A Study in Parallel Algorithms : Stream Compaction

There are two main components of stream compaction: scan and scatter.

Here is a comparison of the various mehtods I used to scan:

![](https://lh4.googleusercontent.com/TWSCNE_ZOLPWiv-EFjObiNwU7AW9Qfz5X4F-wtiu6JngBCe1ZIg_T5HCn5_k8q8d4OnJkageIPI=w1505-h726)

As you can see, the serial version is faster for small arrays, but is quickly out matched as the array length grows.  The global
memory version is always just a bit slower than the shared memory version, which makes sense, as the only difference is the slowdown
that comes from fetching from global memory often.  The work efficient algorithm that I've implemented must have a bug in it, because
it only becomes comparable to the naive shared memory version after the array is over 10 million elements long.  Further investigation is
needed.

And here is a comparison of my scatter implementation and thrust's.  I think I'm using a slow
thrust version of this, becuase I don't think my basic version in CUDA should be as fast as thrust.

![](https://lh3.googleusercontent.com/-smo_LiXzpgg15xhhf7EwXruEdDWJ6cN-NfNbUv0Z9F7l4qwYAyI22eZpwk9dHrYbonYsrSY9ik=w1505-h726)


# REFERENCES
"Parallel Prefix Sum (Scan) with CUDA." GPU Gems 3.
