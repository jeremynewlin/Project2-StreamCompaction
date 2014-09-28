A Study in Parallel Algorithms : Stream Compaction

There are two main components of stream compaction: scan and scatter.

Here is a comparison of the various mehtods I used to scan:

![](http://imgur.com/AaR3gk0,V55kt3w)

As you can see, the serial version is faster for small arrays, but is quickly out matched as the array length grows.  The global
memory version is always just a bit slower than the shared memory version, which makes sense, as the only difference is the slowdown
that comes from fetching from global memory often.  The work efficient algorithm that I've implemented must have a bug in it, because
it only becomes comparable to the naive shared memory version after the array is over 10 million elements long.  Further investigation is
needed.

And here is a comparison of my scatter implementation and thrust's.  I think I'm using a slow
thrust version of this, becuase I don't think my basic version in CUDA should be as fast as thrust.  But, to be honest,
I'm not sure how else to optimize my implementation of scatter any further.  It has 3 global memory reads that are absolutely necessary,
and a branch.

![](http://imgur.com/AaR3gk0,V55kt3w#1)


# REFERENCES
"Parallel Prefix Sum (Scan) with CUDA." GPU Gems 3.
