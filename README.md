# PDF-HR

## Training

To train a model, run the following command:
```
./run.sh 1 args_deepmimic_nrdf/deepmimic_g1_args_backflip.txt
```
Parameters description:

- `use_nrdf_reward`: True  # trigger for nrdf reward 
- `reward_nrdf_dist_w`: 0.2 # trigger for nrdf reward 
- `reward_nrdf_dist_scale`: 1.0
- `reward_nrdf_mode`: "static"  # static or dynamic
- `reward_nrdf_tolerance`: 0.05
- `nrdf_model_path`: "prior_ckpts/nrdf_epoch_70.pt"
- `use_nrdf_early_termination`: False  # default as False
- `nrdf_et_threshold`: 0.4


## Testing

To test a model, run the following command:
```
python mimickit/run.py --arg_file args/deepmimic_humanoid_ppo_args.txt --num_envs 4 --visualize true --mode test --model_file data/models/deepmimic_humanoid_spinkick_model.pt
```
- `--model_file` specifies the `.pt` file that contains the parameters of the trained model. Pretrained models are available in [`data/models/`](data/models/), and the corresponding training log files are available in [`data/logs/`](data/logs/).

