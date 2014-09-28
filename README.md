A Study in Parallel Algorithms : Stream Compaction

There are two main components of stream compaction: scan and scatter.

Here is a comparison of the various mehtods I used to scan:

![](https://drive.google.com/file/d/0BzqFSVys9HdcV0t2eE43YXgydDQ/edit?usp=sharing)

And here is a comparison of my scatter implementation and thrust's.  I think I'm using a slow
thrut version of this, becuase I don't think my basic version in CUDA should be as fast as thrust.

![](https://drive.google.com/file/d/0BzqFSVys9HdcV0t2eE43YXgydDQ/edit?usp=sharing)


# REFERENCES
"Parallel Prefix Sum (Scan) with CUDA." GPU Gems 3.
