# rerank-lite

Small project to test the feasibility of re-ranking poses. Emphasis is on
getting a positive or negative answer to the question "can you learn to re-rank
poses?" as quickly as possible. Transferring results is going to be a challenge
if they are positive, but failing to get any re-ranking performance boost on
artificial data suggests that the actual idea is just silly.

## Data

You'll need to download the
[MPII Human Pose dataset](http://human-pose.mpi-inf.mpg.de/) and stick it in
`data/`. When you're done, `data/images/` should contain all of the images for
MPII Human Pose, and `data/mpii_human_pose_v1_u12_2/` should contain the pose
data itself. You'll have to use `convert/data_to_json.m` to convert
`data/mpii_human_pose_v1_u12_2/mpii_human_pose_v1_u12_1.mat` to a JSON file,
then stick that in `data/mpii_human_pose_v1_u12_1.json`

You'll also need a copy of COCO, which should be symlinked into `data/coco`. On
`paloalto`, `./data/coco` should point to `/data/coco/coco` (which contains
`Images/` and `annotations/` subdirectories).
