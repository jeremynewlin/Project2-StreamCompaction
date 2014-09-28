A Study in Parallel Algorithms : Stream Compaction

There are two main components of stream compaction: scan and scatter.

Here is a comparison of the various mehtods I used to scan:

![](https://lh4.googleusercontent.com/TWSCNE_ZOLPWiv-EFjObiNwU7AW9Qfz5X4F-wtiu6JngBCe1ZIg_T5HCn5_k8q8d4OnJkageIPI=w1505-h726)

And here is a comparison of my scatter implementation and thrust's.  I think I'm using a slow
thrust version of this, becuase I don't think my basic version in CUDA should be as fast as thrust.

![](https://drive.google.com/file/d/0BzqFSVys9HdcV0t2eE43YXgydDQ/edit?usp=sharing)


# REFERENCES
"Parallel Prefix Sum (Scan) with CUDA." GPU Gems 3.
