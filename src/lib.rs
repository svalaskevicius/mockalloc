#![feature(allocator_api)]
#![feature(slice_ptr_get)]
#![deny(missing_docs)]
//! Mockalloc is a crate to allow testing code which uses the global allocator. It
//! uses a probabilistic algorithm to detect and distinguish several kinds of
//! allocation related bugs:
//!
//! - Memory leaks
//! - Double frees
//! - Invalid frees (bad pointer)
//! - Invalid frees (bad size)
//! - Invalid frees (bad alignment)
//!
//! Once a bug is detected, you can enable the `tracing` feature of this crate
//! to collect detailed information about the problem including backtraces showing
//! where memory was allocated and freed.
//!
//! In the case the memory was leaked, it is also possible to find a list of
//! backtraces showing possibilities for where we expected the memory to be freed.
//!
//! Note: the `tracing` feature incurs a significant performance penalty. (Although it
//! is significantly faster than running the code under `miri`). You should also be
//! aware that backtraces are often less complete in release builds where many frames are
//! optimized out.
//!
//! ## Usage
//!
//! Typical use involves enabling the `Mockalloc` allocator during tests, eg:
//!
//! ```rust
//! #[cfg(test)]
//! mod tests {
//!     use std::alloc::System;
//!     use mockalloc::Mockalloc;
//!
//!     #[global_allocator]
//!     static ALLOCATOR: Mockalloc<System> = Mockalloc(System);
//! }
//! ```
//!
//! Once the allocator is enabled, there are several ways to use it in your tests.
//!
//! The easiest way is to use the `#[mockalloc::test]` attribute on your tests
//! instead of the usual `#[test]` attribute:
//!
//! ```rust
//!     #[mockalloc::test]
//!     fn it_works() {
//!         // Some code which uses the allocator
//!     }
//! ```
//!
//! The test will fail if any of the allocation bugs listed above are detected.
//! The test will also fail with the `NoData` error if no allocations are detected
//! so that you can be sure that the `Mockalloc` allocator is active.
//!
//! You can also use `mockalloc` to test a specific section of code for memory
//! issues without checking the entire test using the `assert_allocs` function.
//!
//! The `#[mockalloc::test]` attribute in the prior example is simply a shorthand
//! for:
//!
//! ```rust
//!     #[test]
//!     fn it_works() {
//!         mockalloc::assert_allocs(|| {
//!             // Some code which uses the allocator
//!         });
//!     }
//! ```
//!
//! It is also possible to make more detailed assertions: for example you may want
//! to assert that a piece of code performs a specific number of allocations. For
//! this you can use the `record_allocs` function:
//!
//! ```rust
//!     #[test]
//!     fn it_works() {
//!         let alloc_info = mockalloc::record_allocs(|| {
//!             // Some code which uses the allocator
//!         });
//!
//!         assert_eq!(alloc_info.num_allocs(), 2);
//!
//!         // This is what `assert_allocs` does internally:
//!         alloc_info.result().unwrap()
//!     }
//! ```
//!
//! ## Limitations
//!
//! Allocations are tracked separately for each thread. This allows tests to be
//! run in parallel, but it means that the library will report false positives
//! if a pointer returned by an allocation on one thread is later freed by a
//! different thread.
//!
//! When the `tracing` feature is disabled, the algorithm cannot detect where the
//! bug is, it can only indicate what kind of bug is present.
//!
//! ## How it works
//!
//! The allocator does its tracking without allocating any memory itself. It
//! uses a probabilistic algorithm which works by hashing various pieces of
//! metadata about allocations and frees, and then accumulating these using
//! a commutative operation so that the order does not affect the result.
//!
//! Depending on which of these accumulators returns to zero by the end of
//! a region under test, different allocation bugs can be distinguished.
//!
//! The following metadata is hashed and accumulated:
//!
//! - Pointer
//! - Size & Pointer
//! - Alignment & Pointer
//!
//! In addition to tracking the total number of allocations and frees.
//!
//! We can detect memory leaks and double frees by looking for a difference
//! between the total numbers of allocations and frees.
//!
//! Otherwise, if the pointer accumulator does not return to zero, we know that
//! an invalid pointer was freed.
//!
//! Otherwise, we know the right pointers were freed, but maybe with the wrong
//! size and/or alignment, which we can detect with the other two accumulators.
//!
//! If all accumulators returned to zero then we know everything is good.
//!
//! Each accumulator and hash is 128 bits to essentially eliminate the chance
//! of a collision.

use std::alloc::{Allocator, GlobalAlloc, Layout};
use std::cell::{Cell, RefCell};
use std::thread_local;

#[cfg(feature = "tracing")]
/// Functionality for detailed tracing of allocations. Enabled with the
/// `tracing` feature.
pub mod tracing;

// Probably overkill, but performance isn't a huge concern
fn hash_fn(p: usize) -> u128 {
    const PRIME1: u128 = 257343791756393576901679996513787191589;
    const PRIME2: u128 = 271053192961985756828288246809453504189;
    let mut p = (p as u128).wrapping_add(PRIME2);
    p = p.wrapping_mul(PRIME1);
    p = p ^ (p >> 64);
    p = p.wrapping_mul(PRIME2);
    p = p ^ (p >> 42);
    p = p.wrapping_mul(PRIME1);
    p = p ^ (p >> 25);
    p
}

#[derive(Default)]
struct LocalState {
    ptr_accum: u128,
    ptr_size_accum: u128,
    ptr_align_accum: u128,
    num_allocs: u64,
    num_frees: u64,
    mem_allocated: u64,
    mem_freed: u64,
    peak_mem: u64,
    peak_mem_allocs: u64,
    #[cfg(feature = "tracing")]
    tracing: tracing::TracingState,
}

impl LocalState {
    fn record_alloc(&mut self, ptr: *const u8, layout: Layout) {
        if ptr.is_null() {
            return;
        }
        let ptr_hash = hash_fn(ptr as usize);
        let size_hash = hash_fn(layout.size());
        let align_hash = hash_fn(layout.align());
        self.ptr_accum = self.ptr_accum.wrapping_add(ptr_hash);
        self.ptr_size_accum = self
            .ptr_size_accum
            .wrapping_add(ptr_hash.wrapping_mul(size_hash));
        self.ptr_align_accum = self
            .ptr_align_accum
            .wrapping_add(ptr_hash.wrapping_mul(align_hash));
        self.num_allocs += 1;
        self.mem_allocated += layout.size() as u64;

        if self.mem_allocated > self.mem_freed {
            let mem_usage = self.mem_allocated - self.mem_freed;
            if mem_usage > self.peak_mem {
                self.peak_mem = mem_usage;
                self.peak_mem_allocs = self.num_allocs.saturating_sub(self.num_frees);
            }
        }

        #[cfg(feature = "tracing")]
        self.tracing.record_alloc(ptr, layout);
    }
    fn record_free(&mut self, ptr: *const u8, layout: Layout) {
        let ptr_hash = hash_fn(ptr as usize);
        let size_hash = hash_fn(layout.size());
        let align_hash = hash_fn(layout.align());
        self.ptr_accum = self.ptr_accum.wrapping_sub(ptr_hash);
        self.ptr_size_accum = self
            .ptr_size_accum
            .wrapping_sub(ptr_hash.wrapping_mul(size_hash));
        self.ptr_align_accum = self
            .ptr_align_accum
            .wrapping_sub(ptr_hash.wrapping_mul(align_hash));
        self.num_frees += 1;
        self.mem_freed += layout.size() as u64;

        #[cfg(feature = "tracing")]
        self.tracing.record_free(ptr, layout);
    }
    fn start(&mut self) {
        *self = Default::default();
        #[cfg(feature = "tracing")]
        self.tracing.start();
    }

    fn finish(&mut self) -> AllocInfo {
        let result = if self.num_allocs > self.num_frees {
            Err(AllocError::Leak)
        } else if self.num_allocs < self.num_frees {
            Err(AllocError::DoubleFree)
        } else if self.num_allocs == 0 {
            Err(AllocError::NoData)
        } else if self.ptr_accum != 0 {
            Err(AllocError::BadPtr)
        } else {
            match (self.ptr_size_accum != 0, self.ptr_align_accum != 0) {
                (true, true) => Err(AllocError::BadLayout),
                (true, false) => Err(AllocError::BadSize),
                (false, true) => Err(AllocError::BadAlignment),
                (false, false) => Ok(()),
            }
        };
        AllocInfo {
            result,
            num_allocs: self.num_allocs,
            num_frees: self.num_frees,
            mem_allocated: self.mem_allocated,
            mem_freed: self.mem_freed,
            peak_mem: self.peak_mem,
            peak_mem_allocs: self.peak_mem_allocs,
            #[cfg(feature = "tracing")]
            tracing: self.tracing.finish(),
        }
    }
}

thread_local! {
    static ENABLED: Cell<bool> = Cell::new(false);
    static LOCAL_STATE: RefCell<LocalState> = RefCell::new(LocalState::default());
}

/// Wraps an existing allocator to allow detecting allocation bugs.
/// You should use the `#[global_allocator]` attribute to activate
/// this allocator.
pub struct Mockalloc<T: Allocator>(pub T);

unsafe impl<T: Allocator> Allocator for Mockalloc<T> {
    fn allocate(&self, layout: Layout) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
        let ptr = self.0.allocate(layout);
        with_local_state(|state| {
            state.record_alloc(ptr.unwrap().as_mut_ptr(), layout);
        });
        ptr
    }

    unsafe fn deallocate(&self, ptr: std::ptr::NonNull<u8>, layout: Layout) {
        with_local_state(|state| {
            state.record_free(ptr.as_ptr(), layout);
        });
        self.0.deallocate(ptr, layout);
    }

    fn allocate_zeroed(
        &self,
        layout: Layout,
    ) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
        let ptr = self.0.allocate_zeroed(layout);
        with_local_state(|state| {
            state.record_alloc(ptr.unwrap().as_mut_ptr(), layout);
        });
        ptr
    }

    unsafe fn grow(
        &self,
        ptr: std::ptr::NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
        let new_ptr = self.0.grow(ptr, old_layout, new_layout);
        with_local_state(|state| {
            state.record_free(ptr.as_ptr(), old_layout);
            state.record_alloc(new_ptr.unwrap().as_mut_ptr(), new_layout);
        });
        new_ptr
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: std::ptr::NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
        let new_ptr = self.0.grow_zeroed(ptr, old_layout, new_layout);
        with_local_state(|state| {
            state.record_free(ptr.as_ptr(), old_layout);
            state.record_alloc(new_ptr.unwrap().as_mut_ptr(), new_layout);
        });
        new_ptr
    }

    unsafe fn shrink(
        &self,
        ptr: std::ptr::NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<std::ptr::NonNull<[u8]>, std::alloc::AllocError> {
        let new_ptr = self.0.shrink(ptr, old_layout, new_layout);
        with_local_state(|state| {
            state.record_free(ptr.as_ptr(), old_layout);
            state.record_alloc(new_ptr.unwrap().as_mut_ptr(), new_layout);
        });
        new_ptr
    }
}

/// Types of allocation bug which can be detected by the allocator.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum AllocError {
    /// No allocations were detected. Perhaps `Mockalloc` isn't enabled
    /// as the global allocator?
    NoData,
    /// There were more calls to `alloc` than to `dealloc`.
    Leak,
    /// There were more calls to `dealloc` than to `alloc`.
    DoubleFree,
    /// A pointer was passed to `dealloc` which was not previously
    /// returned by `alloc`.
    BadPtr,
    /// The size specified in a call to `dealloc` did not match that
    /// specified in the corresponding `alloc` call.
    BadSize,
    /// The alignment specified in a call to `dealloc` did not match that
    /// specified in the corresponding `alloc` call.
    BadAlignment,
    /// The size and alignment specified in a call to `dealloc` did not match
    /// those specified in the corresponding `alloc` call.
    BadLayout,
}

/// Captures information about the allocations performed by a region under
/// test.
#[derive(Debug, Clone)]
pub struct AllocInfo {
    num_allocs: u64,
    num_frees: u64,
    mem_allocated: u64,
    mem_freed: u64,
    peak_mem: u64,
    peak_mem_allocs: u64,
    result: Result<(), AllocError>,
    #[cfg(feature = "tracing")]
    tracing: tracing::TracingInfo,
}

impl AllocInfo {
    /// Returns the total number of allocations performed.
    pub fn num_allocs(&self) -> u64 {
        self.num_allocs
    }
    /// Returns the total number of frees performed.
    pub fn num_frees(&self) -> u64 {
        self.num_frees
    }
    /// Returns the total number of frees performed.
    pub fn num_leaks(&self) -> u64 {
        self.num_allocs - self.num_frees
    }
    /// Returns the total amount of memory allocated.
    pub fn mem_allocated(&self) -> u64 {
        self.mem_allocated
    }
    /// Returns the total amount of memory leaked.
    pub fn mem_leaked(&self) -> u64 {
        self.mem_allocated - self.mem_freed
    }
    /// Returns the total amount of memory leaked.
    pub fn mem_freed(&self) -> u64 {
        self.mem_freed
    }
    /// Returns peak memory usage, not including any overhead used by the allocator.
    pub fn peak_mem(&self) -> u64 {
        self.peak_mem
    }
    /// Returns the number of active allocations during peak memory usage.
    pub fn peak_mem_allocs(&self) -> u64 {
        self.peak_mem_allocs
    }
    /// Returns an `Err(..)` result if any allocation bugs were detected.
    pub fn result(&self) -> Result<(), AllocError> {
        self.result.clone()
    }
    /// Returns the detailed trace of leaks and errors.
    #[cfg(feature = "tracing")]
    pub fn tracing(&self) -> &tracing::TracingInfo {
        &self.tracing
    }
}

///
pub struct AllocChecker(bool);

impl AllocChecker {
    ///
    pub fn new() -> Self {
        LOCAL_STATE.with(|rc| rc.borrow_mut().start());
        ENABLED.with(|c| {
            assert!(!c.get(), "Mockalloc already recording");
            c.set(true);
        });
        Self(true)
    }

    ///
    pub fn finish(mut self) -> AllocInfo {
        self.0 = false;
        ENABLED.with(|c| c.set(false));
        LOCAL_STATE.with(|rc| rc.borrow_mut().finish())
    }
}

impl Drop for AllocChecker {
    fn drop(&mut self) {
        if self.0 {
            ENABLED.with(|c| c.set(false));
            LOCAL_STATE.with(|rc| rc.borrow_mut().finish());
        }
    }
}

/// Records the allocations within a code block.
pub fn record_allocs(f: impl FnOnce()) -> AllocInfo {
    let checker = AllocChecker::new();
    f();
    checker.finish()
}

/// Records the allocations within a code block and asserts that no issues
/// were detected.
///
/// No checks are performed if `miri` is detected, as we cannot collect
/// allocation data in that case, and `miri` performs many of these
/// checks already.
///
/// If the `tracing` feature is enabled and an error or leak is detected,
/// this function also prints out the full trace to `stderr`.
pub fn assert_allocs(f: impl FnOnce()) {
    if cfg!(miri) {
        f();
    } else {
        let info = record_allocs(f);
        #[cfg(feature = "tracing")]
        if info.result.is_err() {
            eprintln!("# Mockalloc trace:\n\n{:#?}", info.tracing);
        }
        info.result.unwrap();
    }
}

/// Returns `true` if allocations are currently being recorded, ie. if
/// we're inside a call to `record_allocs`.
pub fn is_recording() -> bool {
    ENABLED.with(|c| c.get())
}

fn with_local_state(f: impl FnOnce(&mut LocalState)) {
    if !is_recording() {
        return;
    }
    ENABLED.with(|c| c.set(false));
    LOCAL_STATE.with(|rc| f(&mut rc.borrow_mut()));
    ENABLED.with(|c| c.set(true));
}

pub use mockalloc_macros::test;

#[cfg(test)]
mod tests {
    use super::{is_recording, record_allocs, AllocError, Mockalloc};
    use std::alloc::{Allocator, Global, GlobalAlloc, Layout, System};
    use std::{cmp, mem, ptr};

    struct LeakingAllocator(Global);

    unsafe impl Allocator for LeakingAllocator {
        fn allocate(&self, layout: Layout) -> Result<ptr::NonNull<[u8]>, std::alloc::AllocError> {
            self.0.allocate(layout)
        }

        unsafe fn deallocate(&self, ptr: ptr::NonNull<u8>, layout: Layout) {
            if !is_recording() {
                self.0.deallocate(ptr, layout);
            }
        }
    }

    // We suppress calls to `dealloc` whilst recording so that our tests don't cause UB
    // when simulating bad requests to the allocator.
    static A: Mockalloc<LeakingAllocator> = Mockalloc(LeakingAllocator(Global));
    type A = &'static Mockalloc<LeakingAllocator>;

    fn do_some_allocations() -> Vec<Box<i32, A>, A> {
        let mut a = Vec::new_in(&A);
        let mut b = Vec::new_in(&A);
        for i in 0..32 {
            let p = Box::new_in(i, &A);
            if i % 2 == 0 {
                a.push(p);
            } else {
                b.push(p);
            }
        }
        a
    }

    #[test]
    fn it_works() {
        let alloc_info = record_allocs(|| {
            let _p = Box::new_in(42, &A);
        });
        alloc_info.result().unwrap();
        assert_eq!(alloc_info.num_allocs(), 1);
        assert_eq!(alloc_info.num_frees(), 1);
        assert_eq!(alloc_info.peak_mem(), 4);
        assert_eq!(alloc_info.peak_mem_allocs(), 1);
    }

    #[test]
    fn it_detects_leak() {
        let alloc_info = record_allocs(|| {
            mem::forget(Box::new_in(42, &A));
        });
        assert_eq!(alloc_info.result().unwrap_err(), AllocError::Leak);
        assert_eq!(alloc_info.num_allocs(), 1);
        assert_eq!(alloc_info.num_frees(), 0);
    }

    #[test]
    fn it_detects_bad_layout() {
        let alloc_info = record_allocs(|| unsafe {
            mem::transmute::<_, Box<f64, A>>(Box::new_in(42u32, &A));
        });
        assert_eq!(alloc_info.result().unwrap_err(), AllocError::BadLayout);
        assert_eq!(alloc_info.num_allocs(), 1);
        assert_eq!(alloc_info.num_frees(), 1);
    }

    #[test]
    fn it_detects_no_data() {
        let alloc_info = record_allocs(|| ());
        assert_eq!(alloc_info.result().unwrap_err(), AllocError::NoData);
        assert_eq!(alloc_info.num_allocs(), 0);
        assert_eq!(alloc_info.num_frees(), 0);
    }

    #[test]
    fn it_detects_bad_alignment() {
        let alloc_info = record_allocs(|| unsafe {
            mem::transmute::<_, Box<[u8; 4], A>>(Box::new_in(42u32, &A));
        });
        assert_eq!(alloc_info.result().unwrap_err(), AllocError::BadAlignment);
        assert_eq!(alloc_info.num_allocs(), 1);
        assert_eq!(alloc_info.num_frees(), 1);
    }

    #[test]
    fn it_detects_bad_size() {
        let alloc_info = record_allocs(|| unsafe {
            mem::transmute::<_, Box<[u32; 2], A>>(Box::new_in(42u32, &A));
        });
        assert_eq!(alloc_info.result().unwrap_err(), AllocError::BadSize);
        assert_eq!(alloc_info.num_allocs(), 1);
        assert_eq!(alloc_info.num_frees(), 1);
    }

    #[test]
    fn it_detects_double_free() {
        let alloc_info = record_allocs(|| unsafe {
            let mut x = mem::ManuallyDrop::new(Box::new_in(42, &A));
            mem::ManuallyDrop::drop(&mut x);
            mem::ManuallyDrop::drop(&mut x);
        });
        assert_eq!(alloc_info.result().unwrap_err(), AllocError::DoubleFree);
        assert_eq!(alloc_info.num_allocs(), 1);
        assert_eq!(alloc_info.num_frees(), 2);
    }

    #[test]
    fn it_detects_bad_ptr() {
        let alloc_info = record_allocs(|| unsafe {
            let mut x = Box::new_in(42, &A);
            *mem::transmute::<_, &mut usize>(&mut x) += 1;
        });
        assert_eq!(alloc_info.result().unwrap_err(), AllocError::BadPtr);
        assert_eq!(alloc_info.num_allocs(), 1);
        assert_eq!(alloc_info.num_frees(), 1);
    }

    #[test]
    fn it_works_amongst_many() {
        let alloc_info = record_allocs(|| {
            let _unused = do_some_allocations();
            let _p = Box::new_in(42, &A);
            let _unused = do_some_allocations();
        });
        alloc_info.result().unwrap();
        assert_eq!(alloc_info.peak_mem(), 964);
        assert_eq!(alloc_info.peak_mem_allocs(), 52);
    }

    #[test]
    fn it_detects_leak_amongst_many() {
        let alloc_info = record_allocs(|| {
            let _unused = do_some_allocations();
            let p = Box::new_in(42, &A);
            let _unused = do_some_allocations();
            mem::forget(p);
            let _unused = do_some_allocations();
        });
        assert_eq!(alloc_info.result().unwrap_err(), AllocError::Leak);
    }

    #[test]
    fn it_detects_bad_layout_amongst_many() {
        let alloc_info = record_allocs(|| unsafe {
            let _unused = do_some_allocations();
            let p = Box::new_in(42u32, &A);
            let _unused = do_some_allocations();
            mem::transmute::<_, Box<f64, A>>(p);
            let _unused = do_some_allocations();
        });
        assert_eq!(alloc_info.result().unwrap_err(), AllocError::BadLayout);
    }

    #[test]
    fn it_detects_bad_alignment_amongst_many() {
        let alloc_info = record_allocs(|| unsafe {
            let _unused = do_some_allocations();
            let p = Box::new_in(42u32, &A);
            let _unused = do_some_allocations();
            mem::transmute::<_, Box<[u8; 4], A>>(p);
            let _unused = do_some_allocations();
        });
        assert_eq!(alloc_info.result().unwrap_err(), AllocError::BadAlignment);
    }

    #[test]
    fn it_detects_bad_size_amongst_many() {
        let alloc_info = record_allocs(|| unsafe {
            let _unused = do_some_allocations();
            let p = Box::new_in(42u32, &A);
            let _unused = do_some_allocations();
            mem::transmute::<_, Box<[u32; 2], A>>(p);
            let _unused = do_some_allocations();
        });
        assert_eq!(alloc_info.result().unwrap_err(), AllocError::BadSize);
    }

    #[test]
    fn it_detects_double_free_amongst_many() {
        let alloc_info = record_allocs(|| unsafe {
            let _unused = do_some_allocations();
            let mut x = mem::ManuallyDrop::new(Box::new_in(42, &A));
            let _unused = do_some_allocations();
            mem::ManuallyDrop::drop(&mut x);
            let _unused = do_some_allocations();
            mem::ManuallyDrop::drop(&mut x);
            let _unused = do_some_allocations();
        });
        assert_eq!(alloc_info.result().unwrap_err(), AllocError::DoubleFree);
    }

    #[test]
    fn it_detects_bad_ptr_amongst_many() {
        let alloc_info = record_allocs(|| unsafe {
            let _unused = do_some_allocations();
            let mut x = Box::new_in(42, &A);
            let _unused = do_some_allocations();
            *mem::transmute::<_, &mut usize>(&mut x) += 1;
            let _unused = do_some_allocations();
        });
        assert_eq!(alloc_info.result().unwrap_err(), AllocError::BadPtr);
    }
}
