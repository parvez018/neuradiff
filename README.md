# Physics Knowledge Discovery via Neural Differential Equation Embedding
Code repository for NeuraDiff paper accepted in ECML-PKDD-2021 for publication
## Abstract
Despite much interest, physics knowledge discovery from experiment data still remains largely a manual trial-and-error process. This paper proposes neural differential equation embedding **NeuraDiff**, an end-to-end approach to learn a physics model characterized by a set of partial differential equations  directly from experiment data. The key idea is the integration of two neural networks -- one recognition net extracting the values of physics model variables from experimental data, and the other neural differential equation net simulating the temporal evolution of the physics model. Learning is completed by matching the outcomes of the two neural networks. We apply **NeuraDiff** to the real-world application of  tracking and learning the physics model of nano-scale crystalline defects in materials under irradiation and high temperature. Experimental results demonstrate that **NeuraDiff**  produces highly accurate tracking results while capturing the correct dynamics of nano-scale defects. 

## Full Paper and Presentation Video Link
[Download the paper here](https://2021.ecmlpkdd.org/wp-content/uploads/2021/07/sub_499.pdf)

[Video presentation available here](https://slideslive.com/38963709)

## How to run
1. Download data from this [dropbox link](https://www.dropbox.com/sh/bgrl2zgtypjc90q/AABl65LC6oca4mp9KVpiO72Ra?dl=0
)
2. Organize dataset in *data* directory
3. Pre-training
   - Run `python video_train_only.py`
   - Output model file: *best_video_train_only_unet_ef8.model*

3. Training
   - Run `python irradiation_train_video.py`
   - Output model files: 
     - *best_ts_model_out.model*
     - *best_unet_model_out.model*

4. Evaluation 
   - Generate the annotations and dynamcis videos
     - `python plot_train_video_annotate.py`
     - `python plot_train_video_dynamics.py`

   - Get pixelwise accuracy
     - `python track_acc.py`

All arguments for running the above commands are already set in code

## Description of the source files
| File | Purpose |
| :--- | :--- |
| video_train_only.py | Pre-train the recognition net | 
| irradiation_train_video.py | Train model |
| irradiation_model.py | The neural phase-field net module |
| unet.py | The recognition net module |
| diff_ops.py | Differential operators |
| plot_train_video_annotate.py| Plot annotate result |
| plot_train_video_dynamics.py | Plot dynamics result |
| track_acc.py | Get the pixelwise tracking accuracy |

## Reference

```
@inproceedings{xue2021physics,
  title={Physics Knowledge Discovery via Neural Differential Equation Embedding},
  author={Xue, Yexiang and Nasim, Md and Zhang, Maosen and Fan, Cuncai and Zhang, Xinghang and El-Azab, Anter},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={118--134},
  year={2021},
  organization={Springer}
}
```


