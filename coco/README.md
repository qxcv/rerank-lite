# COCO person labelling tool

I originally wrote this tool to add binary person/no-person labels to all of the
COCO images. Later, I found out that COCO has extremely reliable person labels
(maybe ~1% people and people-like things in all non-person-labelled images), so
this code is now unnecessary. It should still work, though, when used with the
right dependencies.

Note that `coco.py` was released with COCO (although I trimmed it down a bit to
meet my needs).
