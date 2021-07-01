How to run:
1. Download data at https://www.dropbox.com/sh/bgrl2zgtypjc90q/AABl65LC6oca4mp9KVpiO72Ra?dl=0
Organize dataset in 'data' directory
2. python video_train_only.py
For pre-training, output model file: best_video_train_only_unet_ef8.model

3. python irradiation_train_video.py
output model files: best_ts_model_out.model, best_unet_model_out.model

4. Evaluate: Generate the annotations and dynamcis videos, get pixelwise accuracy:
python plot_train_video_annotate.py
python plot_train_video_dynamics.py
python track_acc.py

Arguments are set in the code

Scripts for synthetic data:
video_train_only.py: pre-train recognition net: 
irradiation_train_video.py: train model

irradiation_model.py: the neural phase-field net module
unet.py: the recognition net module
diff_ops.py: differential operators

plot_train_video_annotate.py: plot annotate result
plot_train_video_dynamics.py: plot dynamics result
track_acc.py: get the tracking accuracy


