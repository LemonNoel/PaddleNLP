[33m[2022-08-31 15:38:16,773] [ WARNING][0m - evaluation_strategy reset to IntervalStrategy.STEPS for do_eval is True. you can also set evaluation_strategy='epoch'.[0m
[32m[2022-08-31 15:38:16,773] [    INFO][0m - The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).[0m
[32m[2022-08-31 15:38:16,773] [    INFO][0m - ============================================================[0m
[32m[2022-08-31 15:38:16,773] [    INFO][0m -      Model Configuration Arguments      [0m
[32m[2022-08-31 15:38:16,773] [    INFO][0m - paddle commit id              :65f388690048d0965ec7d3b43fb6ee9d8c6dee7c[0m
[32m[2022-08-31 15:38:16,773] [    INFO][0m - do_save                       :True[0m
[32m[2022-08-31 15:38:16,773] [    INFO][0m - do_test                       :True[0m
[32m[2022-08-31 15:38:16,774] [    INFO][0m - early_stop_patience           :4[0m
[32m[2022-08-31 15:38:16,774] [    INFO][0m - export_type                   :paddle[0m
[32m[2022-08-31 15:38:16,774] [    INFO][0m - model_name_or_path            :ernie-3.0-base-zh[0m
[32m[2022-08-31 15:38:16,774] [    INFO][0m - [0m
[32m[2022-08-31 15:38:16,774] [    INFO][0m - ============================================================[0m
[32m[2022-08-31 15:38:16,774] [    INFO][0m -       Data Configuration Arguments      [0m
[32m[2022-08-31 15:38:16,774] [    INFO][0m - paddle commit id              :65f388690048d0965ec7d3b43fb6ee9d8c6dee7c[0m
[32m[2022-08-31 15:38:16,774] [    INFO][0m - encoder_hidden_size           :200[0m
[32m[2022-08-31 15:38:16,774] [    INFO][0m - prompt                        :“{'text':'text_a'}”和“{'text':'text_b'}”之间的逻辑关系是{'mask'}{'mask'}。[0m
[32m[2022-08-31 15:38:16,774] [    INFO][0m - soft_encoder                  :lstm[0m
[32m[2022-08-31 15:38:16,774] [    INFO][0m - split_id                      :few_all[0m
[32m[2022-08-31 15:38:16,774] [    INFO][0m - task_name                     :bustm[0m
[32m[2022-08-31 15:38:16,775] [    INFO][0m - [0m
[32m[2022-08-31 15:38:16,775] [    INFO][0m - Already cached /ssd2/wanghuijuan03/.paddlenlp/models/ernie-3.0-base-zh/ernie_3.0_base_zh.pdparams[0m
W0831 15:38:16.776784 80128 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 11.2
W0831 15:38:16.780802 80128 gpu_resources.cc:91] device: 0, cuDNN Version: 8.1.
[32m[2022-08-31 15:38:19,751] [    INFO][0m - Already cached /ssd2/wanghuijuan03/.paddlenlp/models/ernie-3.0-base-zh/ernie_3.0_base_zh_vocab.txt[0m
[32m[2022-08-31 15:38:19,776] [    INFO][0m - tokenizer config file saved in /ssd2/wanghuijuan03/.paddlenlp/models/ernie-3.0-base-zh/tokenizer_config.json[0m
[32m[2022-08-31 15:38:19,776] [    INFO][0m - Special tokens file saved in /ssd2/wanghuijuan03/.paddlenlp/models/ernie-3.0-base-zh/special_tokens_map.json[0m
[32m[2022-08-31 15:38:19,777] [    INFO][0m - Using template: [{'add_prefix_space': '', 'hard': '“'}, {'add_prefix_space': '', 'text': 'text_a'}, {'add_prefix_space': '', 'hard': '”和“'}, {'add_prefix_space': '', 'text': 'text_b'}, {'add_prefix_space': '', 'hard': '”之间的逻辑关系是'}, {'add_prefix_space': '', 'mask': None}, {'add_prefix_space': '', 'mask': None}, {'add_prefix_space': '', 'hard': '。'}][0m
[32m[2022-08-31 15:38:19,780] [    INFO][0m - {'0': 0, '1': 1}[0m
2022-08-31 15:38:19,781 INFO [download.py:119] unique_endpoints {''}
[32m[2022-08-31 15:38:20,950] [    INFO][0m - ============================================================[0m
[32m[2022-08-31 15:38:20,950] [    INFO][0m -     Training Configuration Arguments    [0m
[32m[2022-08-31 15:38:20,950] [    INFO][0m - paddle commit id              :65f388690048d0965ec7d3b43fb6ee9d8c6dee7c[0m
[32m[2022-08-31 15:38:20,950] [    INFO][0m - _no_sync_in_gradient_accumulation:True[0m
[32m[2022-08-31 15:38:20,950] [    INFO][0m - adam_beta1                    :0.9[0m
[32m[2022-08-31 15:38:20,950] [    INFO][0m - adam_beta2                    :0.999[0m
[32m[2022-08-31 15:38:20,951] [    INFO][0m - adam_epsilon                  :1e-08[0m
[32m[2022-08-31 15:38:20,951] [    INFO][0m - alpha_rdrop                   :5.0[0m
[32m[2022-08-31 15:38:20,951] [    INFO][0m - alpha_rgl                     :0.5[0m
[32m[2022-08-31 15:38:20,951] [    INFO][0m - current_device                :gpu:0[0m
[32m[2022-08-31 15:38:20,951] [    INFO][0m - dataloader_drop_last          :False[0m
[32m[2022-08-31 15:38:20,951] [    INFO][0m - dataloader_num_workers        :0[0m
[32m[2022-08-31 15:38:20,951] [    INFO][0m - device                        :gpu[0m
[32m[2022-08-31 15:38:20,951] [    INFO][0m - disable_tqdm                  :True[0m
[32m[2022-08-31 15:38:20,951] [    INFO][0m - do_eval                       :True[0m
[32m[2022-08-31 15:38:20,951] [    INFO][0m - do_export                     :False[0m
[32m[2022-08-31 15:38:20,951] [    INFO][0m - do_predict                    :True[0m
[32m[2022-08-31 15:38:20,951] [    INFO][0m - do_train                      :True[0m
[32m[2022-08-31 15:38:20,951] [    INFO][0m - eval_batch_size               :64[0m
[32m[2022-08-31 15:38:20,952] [    INFO][0m - eval_steps                    :100[0m
[32m[2022-08-31 15:38:20,952] [    INFO][0m - evaluation_strategy           :IntervalStrategy.STEPS[0m
[32m[2022-08-31 15:38:20,952] [    INFO][0m - first_max_length              :None[0m
[32m[2022-08-31 15:38:20,952] [    INFO][0m - fp16                          :False[0m
[32m[2022-08-31 15:38:20,952] [    INFO][0m - fp16_opt_level                :O1[0m
[32m[2022-08-31 15:38:20,952] [    INFO][0m - freeze_dropout                :False[0m
[32m[2022-08-31 15:38:20,952] [    INFO][0m - freeze_plm                    :False[0m
[32m[2022-08-31 15:38:20,952] [    INFO][0m - gradient_accumulation_steps   :1[0m
[32m[2022-08-31 15:38:20,952] [    INFO][0m - greater_is_better             :True[0m
[32m[2022-08-31 15:38:20,952] [    INFO][0m - ignore_data_skip              :False[0m
[32m[2022-08-31 15:38:20,952] [    INFO][0m - label_names                   :None[0m
[32m[2022-08-31 15:38:20,952] [    INFO][0m - learning_rate                 :3e-06[0m
[32m[2022-08-31 15:38:20,952] [    INFO][0m - load_best_model_at_end        :True[0m
[32m[2022-08-31 15:38:20,953] [    INFO][0m - local_process_index           :0[0m
[32m[2022-08-31 15:38:20,953] [    INFO][0m - local_rank                    :-1[0m
[32m[2022-08-31 15:38:20,953] [    INFO][0m - log_level                     :-1[0m
[32m[2022-08-31 15:38:20,953] [    INFO][0m - log_level_replica             :-1[0m
[32m[2022-08-31 15:38:20,953] [    INFO][0m - log_on_each_node              :True[0m
[32m[2022-08-31 15:38:20,953] [    INFO][0m - logging_dir                   :./checkpoints/runs/Aug31_15-38-16_instance-3bwob41y-01[0m
[32m[2022-08-31 15:38:20,953] [    INFO][0m - logging_first_step            :False[0m
[32m[2022-08-31 15:38:20,953] [    INFO][0m - logging_steps                 :10[0m
[32m[2022-08-31 15:38:20,953] [    INFO][0m - logging_strategy              :IntervalStrategy.STEPS[0m
[32m[2022-08-31 15:38:20,953] [    INFO][0m - lr_scheduler_type             :SchedulerType.LINEAR[0m
[32m[2022-08-31 15:38:20,953] [    INFO][0m - max_grad_norm                 :1.0[0m
[32m[2022-08-31 15:38:20,953] [    INFO][0m - max_seq_length                :128[0m
[32m[2022-08-31 15:38:20,953] [    INFO][0m - max_steps                     :-1[0m
[32m[2022-08-31 15:38:20,953] [    INFO][0m - metric_for_best_model         :accuracy[0m
[32m[2022-08-31 15:38:20,954] [    INFO][0m - minimum_eval_times            :None[0m
[32m[2022-08-31 15:38:20,954] [    INFO][0m - no_cuda                       :False[0m
[32m[2022-08-31 15:38:20,954] [    INFO][0m - num_train_epochs              :50.0[0m
[32m[2022-08-31 15:38:20,954] [    INFO][0m - optim                         :OptimizerNames.ADAMW[0m
[32m[2022-08-31 15:38:20,954] [    INFO][0m - other_max_length              :None[0m
[32m[2022-08-31 15:38:20,954] [    INFO][0m - output_dir                    :./checkpoints/[0m
[32m[2022-08-31 15:38:20,954] [    INFO][0m - overwrite_output_dir          :False[0m
[32m[2022-08-31 15:38:20,954] [    INFO][0m - past_index                    :-1[0m
[32m[2022-08-31 15:38:20,954] [    INFO][0m - per_device_eval_batch_size    :64[0m
[32m[2022-08-31 15:38:20,954] [    INFO][0m - per_device_train_batch_size   :8[0m
[32m[2022-08-31 15:38:20,954] [    INFO][0m - ppt_adam_beta1                :0.9[0m
[32m[2022-08-31 15:38:20,954] [    INFO][0m - ppt_adam_beta2                :0.999[0m
[32m[2022-08-31 15:38:20,954] [    INFO][0m - ppt_adam_epsilon              :1e-08[0m
[32m[2022-08-31 15:38:20,955] [    INFO][0m - ppt_learning_rate             :3e-05[0m
[32m[2022-08-31 15:38:20,955] [    INFO][0m - ppt_weight_decay              :0.0[0m
[32m[2022-08-31 15:38:20,955] [    INFO][0m - prediction_loss_only          :False[0m
[32m[2022-08-31 15:38:20,955] [    INFO][0m - process_index                 :0[0m
[32m[2022-08-31 15:38:20,955] [    INFO][0m - remove_unused_columns         :True[0m
[32m[2022-08-31 15:38:20,955] [    INFO][0m - report_to                     :['visualdl'][0m
[32m[2022-08-31 15:38:20,955] [    INFO][0m - resume_from_checkpoint        :None[0m
[32m[2022-08-31 15:38:20,955] [    INFO][0m - run_name                      :./checkpoints/[0m
[32m[2022-08-31 15:38:20,955] [    INFO][0m - save_on_each_node             :False[0m
[32m[2022-08-31 15:38:20,955] [    INFO][0m - save_steps                    :100[0m
[32m[2022-08-31 15:38:20,955] [    INFO][0m - save_strategy                 :IntervalStrategy.STEPS[0m
[32m[2022-08-31 15:38:20,955] [    INFO][0m - save_total_limit              :None[0m
[32m[2022-08-31 15:38:20,955] [    INFO][0m - scale_loss                    :32768[0m
[32m[2022-08-31 15:38:20,955] [    INFO][0m - seed                          :42[0m
[32m[2022-08-31 15:38:20,956] [    INFO][0m - should_log                    :True[0m
[32m[2022-08-31 15:38:20,956] [    INFO][0m - should_save                   :True[0m
[32m[2022-08-31 15:38:20,956] [    INFO][0m - task_type                     :multi-class[0m
[32m[2022-08-31 15:38:20,956] [    INFO][0m - train_batch_size              :8[0m
[32m[2022-08-31 15:38:20,956] [    INFO][0m - truncate_mode                 :tail[0m
[32m[2022-08-31 15:38:20,956] [    INFO][0m - use_rdrop                     :False[0m
[32m[2022-08-31 15:38:20,956] [    INFO][0m - use_rgl                       :False[0m
[32m[2022-08-31 15:38:20,956] [    INFO][0m - warmup_ratio                  :0.0[0m
[32m[2022-08-31 15:38:20,956] [    INFO][0m - warmup_steps                  :0[0m
[32m[2022-08-31 15:38:20,956] [    INFO][0m - weight_decay                  :0.0[0m
[32m[2022-08-31 15:38:20,956] [    INFO][0m - world_size                    :1[0m
[32m[2022-08-31 15:38:20,956] [    INFO][0m - [0m
[32m[2022-08-31 15:38:20,958] [    INFO][0m - ***** Running training *****[0m
[32m[2022-08-31 15:38:20,958] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 15:38:20,958] [    INFO][0m -   Num Epochs = 50[0m
[32m[2022-08-31 15:38:20,958] [    INFO][0m -   Instantaneous batch size per device = 8[0m
[32m[2022-08-31 15:38:20,958] [    INFO][0m -   Total train batch size (w. parallel, distributed & accumulation) = 8[0m
[32m[2022-08-31 15:38:20,958] [    INFO][0m -   Gradient Accumulation steps = 1[0m
[32m[2022-08-31 15:38:20,958] [    INFO][0m -   Total optimization steps = 1000.0[0m
[32m[2022-08-31 15:38:20,959] [    INFO][0m -   Total num train samples = 8000[0m
[32m[2022-08-31 15:38:22,994] [    INFO][0m - loss: 0.91882019, learning_rate: 2.97e-06, global_step: 10, interval_runtime: 2.0339, interval_samples_per_second: 3.933, interval_steps_per_second: 4.917, epoch: 0.5[0m
[32m[2022-08-31 15:38:23,870] [    INFO][0m - loss: 0.60569806, learning_rate: 2.9400000000000002e-06, global_step: 20, interval_runtime: 0.8771, interval_samples_per_second: 9.121, interval_steps_per_second: 11.401, epoch: 1.0[0m
[32m[2022-08-31 15:38:24,823] [    INFO][0m - loss: 0.53142138, learning_rate: 2.91e-06, global_step: 30, interval_runtime: 0.9523, interval_samples_per_second: 8.401, interval_steps_per_second: 10.501, epoch: 1.5[0m
[32m[2022-08-31 15:38:25,697] [    INFO][0m - loss: 0.55912471, learning_rate: 2.88e-06, global_step: 40, interval_runtime: 0.8746, interval_samples_per_second: 9.147, interval_steps_per_second: 11.434, epoch: 2.0[0m
[32m[2022-08-31 15:38:26,668] [    INFO][0m - loss: 0.50554571, learning_rate: 2.85e-06, global_step: 50, interval_runtime: 0.97, interval_samples_per_second: 8.247, interval_steps_per_second: 10.309, epoch: 2.5[0m
[32m[2022-08-31 15:38:27,542] [    INFO][0m - loss: 0.51269374, learning_rate: 2.82e-06, global_step: 60, interval_runtime: 0.8752, interval_samples_per_second: 9.141, interval_steps_per_second: 11.426, epoch: 3.0[0m
[32m[2022-08-31 15:38:28,480] [    INFO][0m - loss: 0.47652698, learning_rate: 2.7900000000000004e-06, global_step: 70, interval_runtime: 0.937, interval_samples_per_second: 8.538, interval_steps_per_second: 10.672, epoch: 3.5[0m
[32m[2022-08-31 15:38:29,373] [    INFO][0m - loss: 0.44370518, learning_rate: 2.7600000000000003e-06, global_step: 80, interval_runtime: 0.8924, interval_samples_per_second: 8.964, interval_steps_per_second: 11.205, epoch: 4.0[0m
[32m[2022-08-31 15:38:30,322] [    INFO][0m - loss: 0.45279222, learning_rate: 2.73e-06, global_step: 90, interval_runtime: 0.9499, interval_samples_per_second: 8.422, interval_steps_per_second: 10.527, epoch: 4.5[0m
[32m[2022-08-31 15:38:31,225] [    INFO][0m - loss: 0.40981622, learning_rate: 2.7e-06, global_step: 100, interval_runtime: 0.9034, interval_samples_per_second: 8.856, interval_steps_per_second: 11.07, epoch: 5.0[0m
[32m[2022-08-31 15:38:31,226] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 15:38:31,226] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 15:38:31,226] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:38:31,226] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:38:31,226] [    INFO][0m -   Total prediction steps = 3[0m
[32m[2022-08-31 15:38:31,897] [    INFO][0m - eval_loss: 0.526604950428009, eval_accuracy: 0.725, eval_runtime: 0.6702, eval_samples_per_second: 238.743, eval_steps_per_second: 4.476, epoch: 5.0[0m
[32m[2022-08-31 15:38:31,897] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-100[0m
[32m[2022-08-31 15:38:31,897] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:38:35,335] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-100/tokenizer_config.json[0m
[32m[2022-08-31 15:38:35,336] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-100/special_tokens_map.json[0m
[32m[2022-08-31 15:38:41,859] [    INFO][0m - loss: 0.43020835, learning_rate: 2.6700000000000003e-06, global_step: 110, interval_runtime: 10.634, interval_samples_per_second: 0.752, interval_steps_per_second: 0.94, epoch: 5.5[0m
[32m[2022-08-31 15:38:42,732] [    INFO][0m - loss: 0.38311577, learning_rate: 2.64e-06, global_step: 120, interval_runtime: 0.8722, interval_samples_per_second: 9.172, interval_steps_per_second: 11.465, epoch: 6.0[0m
[32m[2022-08-31 15:38:43,657] [    INFO][0m - loss: 0.43324738, learning_rate: 2.61e-06, global_step: 130, interval_runtime: 0.9254, interval_samples_per_second: 8.645, interval_steps_per_second: 10.806, epoch: 6.5[0m
[32m[2022-08-31 15:38:44,531] [    INFO][0m - loss: 0.31333919, learning_rate: 2.58e-06, global_step: 140, interval_runtime: 0.874, interval_samples_per_second: 9.153, interval_steps_per_second: 11.442, epoch: 7.0[0m
[32m[2022-08-31 15:38:45,456] [    INFO][0m - loss: 0.39105437, learning_rate: 2.55e-06, global_step: 150, interval_runtime: 0.9249, interval_samples_per_second: 8.649, interval_steps_per_second: 10.811, epoch: 7.5[0m
[32m[2022-08-31 15:38:46,338] [    INFO][0m - loss: 0.32381623, learning_rate: 2.52e-06, global_step: 160, interval_runtime: 0.8824, interval_samples_per_second: 9.067, interval_steps_per_second: 11.333, epoch: 8.0[0m
[32m[2022-08-31 15:38:47,263] [    INFO][0m - loss: 0.29562595, learning_rate: 2.49e-06, global_step: 170, interval_runtime: 0.925, interval_samples_per_second: 8.649, interval_steps_per_second: 10.811, epoch: 8.5[0m
[32m[2022-08-31 15:38:48,140] [    INFO][0m - loss: 0.29954891, learning_rate: 2.4599999999999997e-06, global_step: 180, interval_runtime: 0.8771, interval_samples_per_second: 9.121, interval_steps_per_second: 11.401, epoch: 9.0[0m
[32m[2022-08-31 15:38:49,072] [    INFO][0m - loss: 0.28225102, learning_rate: 2.43e-06, global_step: 190, interval_runtime: 0.931, interval_samples_per_second: 8.593, interval_steps_per_second: 10.741, epoch: 9.5[0m
[32m[2022-08-31 15:38:49,952] [    INFO][0m - loss: 0.22393932, learning_rate: 2.4000000000000003e-06, global_step: 200, interval_runtime: 0.8809, interval_samples_per_second: 9.081, interval_steps_per_second: 11.351, epoch: 10.0[0m
[32m[2022-08-31 15:38:49,953] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 15:38:49,953] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 15:38:49,953] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:38:49,953] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:38:49,953] [    INFO][0m -   Total prediction steps = 3[0m
[32m[2022-08-31 15:38:50,626] [    INFO][0m - eval_loss: 0.578708291053772, eval_accuracy: 0.75625, eval_runtime: 0.6717, eval_samples_per_second: 238.19, eval_steps_per_second: 4.466, epoch: 10.0[0m
[32m[2022-08-31 15:38:50,626] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-200[0m
[32m[2022-08-31 15:38:50,627] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:38:53,614] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-200/tokenizer_config.json[0m
[32m[2022-08-31 15:38:53,614] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-200/special_tokens_map.json[0m
[32m[2022-08-31 15:38:59,527] [    INFO][0m - loss: 0.20894065, learning_rate: 2.37e-06, global_step: 210, interval_runtime: 9.5748, interval_samples_per_second: 0.836, interval_steps_per_second: 1.044, epoch: 10.5[0m
[32m[2022-08-31 15:39:00,405] [    INFO][0m - loss: 0.28281405, learning_rate: 2.34e-06, global_step: 220, interval_runtime: 0.8781, interval_samples_per_second: 9.111, interval_steps_per_second: 11.388, epoch: 11.0[0m
[32m[2022-08-31 15:39:01,329] [    INFO][0m - loss: 0.21502042, learning_rate: 2.31e-06, global_step: 230, interval_runtime: 0.9238, interval_samples_per_second: 8.66, interval_steps_per_second: 10.825, epoch: 11.5[0m
[32m[2022-08-31 15:39:02,206] [    INFO][0m - loss: 0.20935082, learning_rate: 2.28e-06, global_step: 240, interval_runtime: 0.8766, interval_samples_per_second: 9.127, interval_steps_per_second: 11.408, epoch: 12.0[0m
[32m[2022-08-31 15:39:03,131] [    INFO][0m - loss: 0.26287067, learning_rate: 2.25e-06, global_step: 250, interval_runtime: 0.9252, interval_samples_per_second: 8.646, interval_steps_per_second: 10.808, epoch: 12.5[0m
[32m[2022-08-31 15:39:04,009] [    INFO][0m - loss: 0.20844803, learning_rate: 2.22e-06, global_step: 260, interval_runtime: 0.8785, interval_samples_per_second: 9.107, interval_steps_per_second: 11.383, epoch: 13.0[0m
[32m[2022-08-31 15:39:04,934] [    INFO][0m - loss: 0.26451156, learning_rate: 2.19e-06, global_step: 270, interval_runtime: 0.9251, interval_samples_per_second: 8.648, interval_steps_per_second: 10.81, epoch: 13.5[0m
[32m[2022-08-31 15:39:05,817] [    INFO][0m - loss: 0.14913199, learning_rate: 2.16e-06, global_step: 280, interval_runtime: 0.8824, interval_samples_per_second: 9.067, interval_steps_per_second: 11.333, epoch: 14.0[0m
[32m[2022-08-31 15:39:06,741] [    INFO][0m - loss: 0.21505184, learning_rate: 2.13e-06, global_step: 290, interval_runtime: 0.9247, interval_samples_per_second: 8.652, interval_steps_per_second: 10.815, epoch: 14.5[0m
[32m[2022-08-31 15:39:07,623] [    INFO][0m - loss: 0.14057403, learning_rate: 2.1e-06, global_step: 300, interval_runtime: 0.882, interval_samples_per_second: 9.07, interval_steps_per_second: 11.338, epoch: 15.0[0m
[32m[2022-08-31 15:39:07,624] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 15:39:07,624] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 15:39:07,624] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:39:07,624] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:39:07,624] [    INFO][0m -   Total prediction steps = 3[0m
[32m[2022-08-31 15:39:08,295] [    INFO][0m - eval_loss: 0.6952170133590698, eval_accuracy: 0.75625, eval_runtime: 0.67, eval_samples_per_second: 238.813, eval_steps_per_second: 4.478, epoch: 15.0[0m
[32m[2022-08-31 15:39:08,295] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-300[0m
[32m[2022-08-31 15:39:08,296] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:39:11,258] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-300/tokenizer_config.json[0m
[32m[2022-08-31 15:39:11,259] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-300/special_tokens_map.json[0m
[32m[2022-08-31 15:39:17,440] [    INFO][0m - loss: 0.17382634, learning_rate: 2.07e-06, global_step: 310, interval_runtime: 9.816, interval_samples_per_second: 0.815, interval_steps_per_second: 1.019, epoch: 15.5[0m
[32m[2022-08-31 15:39:18,319] [    INFO][0m - loss: 0.1921545, learning_rate: 2.0400000000000004e-06, global_step: 320, interval_runtime: 0.8792, interval_samples_per_second: 9.099, interval_steps_per_second: 11.373, epoch: 16.0[0m
[32m[2022-08-31 15:39:19,279] [    INFO][0m - loss: 0.1778358, learning_rate: 2.0100000000000002e-06, global_step: 330, interval_runtime: 0.9603, interval_samples_per_second: 8.331, interval_steps_per_second: 10.414, epoch: 16.5[0m
[32m[2022-08-31 15:39:20,162] [    INFO][0m - loss: 0.06417523, learning_rate: 1.98e-06, global_step: 340, interval_runtime: 0.8827, interval_samples_per_second: 9.063, interval_steps_per_second: 11.329, epoch: 17.0[0m
[32m[2022-08-31 15:39:21,088] [    INFO][0m - loss: 0.1283643, learning_rate: 1.95e-06, global_step: 350, interval_runtime: 0.9265, interval_samples_per_second: 8.635, interval_steps_per_second: 10.794, epoch: 17.5[0m
[32m[2022-08-31 15:39:21,965] [    INFO][0m - loss: 0.10683916, learning_rate: 1.9200000000000003e-06, global_step: 360, interval_runtime: 0.8766, interval_samples_per_second: 9.126, interval_steps_per_second: 11.408, epoch: 18.0[0m
[32m[2022-08-31 15:39:22,890] [    INFO][0m - loss: 0.21944242, learning_rate: 1.8900000000000001e-06, global_step: 370, interval_runtime: 0.9251, interval_samples_per_second: 8.647, interval_steps_per_second: 10.809, epoch: 18.5[0m
[32m[2022-08-31 15:39:23,769] [    INFO][0m - loss: 0.08183926, learning_rate: 1.86e-06, global_step: 380, interval_runtime: 0.8791, interval_samples_per_second: 9.1, interval_steps_per_second: 11.375, epoch: 19.0[0m
[32m[2022-08-31 15:39:24,715] [    INFO][0m - loss: 0.08902547, learning_rate: 1.83e-06, global_step: 390, interval_runtime: 0.9455, interval_samples_per_second: 8.461, interval_steps_per_second: 10.576, epoch: 19.5[0m
[32m[2022-08-31 15:39:25,594] [    INFO][0m - loss: 0.11464314, learning_rate: 1.8e-06, global_step: 400, interval_runtime: 0.8788, interval_samples_per_second: 9.103, interval_steps_per_second: 11.379, epoch: 20.0[0m
[32m[2022-08-31 15:39:25,594] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 15:39:25,594] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 15:39:25,595] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:39:25,595] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:39:25,595] [    INFO][0m -   Total prediction steps = 3[0m
[32m[2022-08-31 15:39:26,266] [    INFO][0m - eval_loss: 0.9587444067001343, eval_accuracy: 0.75, eval_runtime: 0.6708, eval_samples_per_second: 238.532, eval_steps_per_second: 4.472, epoch: 20.0[0m
[32m[2022-08-31 15:39:26,266] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-400[0m
[32m[2022-08-31 15:39:26,266] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:39:29,251] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-400/tokenizer_config.json[0m
[32m[2022-08-31 15:39:29,588] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-400/special_tokens_map.json[0m
[32m[2022-08-31 15:39:36,089] [    INFO][0m - loss: 0.1171298, learning_rate: 1.77e-06, global_step: 410, interval_runtime: 10.4958, interval_samples_per_second: 0.762, interval_steps_per_second: 0.953, epoch: 20.5[0m
[32m[2022-08-31 15:39:36,965] [    INFO][0m - loss: 0.15076939, learning_rate: 1.7399999999999999e-06, global_step: 420, interval_runtime: 0.8758, interval_samples_per_second: 9.135, interval_steps_per_second: 11.418, epoch: 21.0[0m
[32m[2022-08-31 15:39:37,889] [    INFO][0m - loss: 0.07621834, learning_rate: 1.71e-06, global_step: 430, interval_runtime: 0.9239, interval_samples_per_second: 8.659, interval_steps_per_second: 10.824, epoch: 21.5[0m
[32m[2022-08-31 15:39:38,767] [    INFO][0m - loss: 0.11129869, learning_rate: 1.6800000000000002e-06, global_step: 440, interval_runtime: 0.8777, interval_samples_per_second: 9.115, interval_steps_per_second: 11.394, epoch: 22.0[0m
[32m[2022-08-31 15:39:39,689] [    INFO][0m - loss: 0.03412145, learning_rate: 1.65e-06, global_step: 450, interval_runtime: 0.9224, interval_samples_per_second: 8.673, interval_steps_per_second: 10.841, epoch: 22.5[0m
[32m[2022-08-31 15:39:40,566] [    INFO][0m - loss: 0.16028023, learning_rate: 1.6200000000000002e-06, global_step: 460, interval_runtime: 0.8766, interval_samples_per_second: 9.127, interval_steps_per_second: 11.408, epoch: 23.0[0m
[32m[2022-08-31 15:39:41,495] [    INFO][0m - loss: 0.14831309, learning_rate: 1.59e-06, global_step: 470, interval_runtime: 0.9293, interval_samples_per_second: 8.609, interval_steps_per_second: 10.761, epoch: 23.5[0m
[32m[2022-08-31 15:39:42,372] [    INFO][0m - loss: 0.01187761, learning_rate: 1.56e-06, global_step: 480, interval_runtime: 0.8771, interval_samples_per_second: 9.121, interval_steps_per_second: 11.402, epoch: 24.0[0m
[32m[2022-08-31 15:39:43,312] [    INFO][0m - loss: 0.16543493, learning_rate: 1.53e-06, global_step: 490, interval_runtime: 0.9405, interval_samples_per_second: 8.506, interval_steps_per_second: 10.632, epoch: 24.5[0m
[32m[2022-08-31 15:39:44,197] [    INFO][0m - loss: 0.02590601, learning_rate: 1.5e-06, global_step: 500, interval_runtime: 0.8849, interval_samples_per_second: 9.04, interval_steps_per_second: 11.301, epoch: 25.0[0m
[32m[2022-08-31 15:39:44,198] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 15:39:44,198] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 15:39:44,198] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:39:44,198] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:39:44,198] [    INFO][0m -   Total prediction steps = 3[0m
[32m[2022-08-31 15:39:44,865] [    INFO][0m - eval_loss: 1.18658447265625, eval_accuracy: 0.775, eval_runtime: 0.6668, eval_samples_per_second: 239.965, eval_steps_per_second: 4.499, epoch: 25.0[0m
[32m[2022-08-31 15:39:44,866] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-500[0m
[32m[2022-08-31 15:39:44,866] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:39:47,960] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-500/tokenizer_config.json[0m
[32m[2022-08-31 15:39:47,961] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-500/special_tokens_map.json[0m
[32m[2022-08-31 15:39:54,748] [    INFO][0m - loss: 0.12769269, learning_rate: 1.4700000000000001e-06, global_step: 510, interval_runtime: 10.5498, interval_samples_per_second: 0.758, interval_steps_per_second: 0.948, epoch: 25.5[0m
[32m[2022-08-31 15:39:55,636] [    INFO][0m - loss: 0.05282363, learning_rate: 1.44e-06, global_step: 520, interval_runtime: 0.8891, interval_samples_per_second: 8.998, interval_steps_per_second: 11.248, epoch: 26.0[0m
[32m[2022-08-31 15:39:56,598] [    INFO][0m - loss: 0.04817574, learning_rate: 1.41e-06, global_step: 530, interval_runtime: 0.9598, interval_samples_per_second: 8.335, interval_steps_per_second: 10.418, epoch: 26.5[0m
[32m[2022-08-31 15:39:57,482] [    INFO][0m - loss: 0.15469859, learning_rate: 1.3800000000000001e-06, global_step: 540, interval_runtime: 0.8861, interval_samples_per_second: 9.028, interval_steps_per_second: 11.285, epoch: 27.0[0m
[32m[2022-08-31 15:39:58,409] [    INFO][0m - loss: 0.04015095, learning_rate: 1.35e-06, global_step: 550, interval_runtime: 0.926, interval_samples_per_second: 8.639, interval_steps_per_second: 10.799, epoch: 27.5[0m
[32m[2022-08-31 15:39:59,310] [    INFO][0m - loss: 0.09321545, learning_rate: 1.32e-06, global_step: 560, interval_runtime: 0.9019, interval_samples_per_second: 8.871, interval_steps_per_second: 11.088, epoch: 28.0[0m
[32m[2022-08-31 15:40:00,246] [    INFO][0m - loss: 0.12601182, learning_rate: 1.29e-06, global_step: 570, interval_runtime: 0.9361, interval_samples_per_second: 8.546, interval_steps_per_second: 10.682, epoch: 28.5[0m
[32m[2022-08-31 15:40:01,129] [    INFO][0m - loss: 0.01643143, learning_rate: 1.26e-06, global_step: 580, interval_runtime: 0.8825, interval_samples_per_second: 9.065, interval_steps_per_second: 11.331, epoch: 29.0[0m
[32m[2022-08-31 15:40:02,056] [    INFO][0m - loss: 0.08194968, learning_rate: 1.2299999999999999e-06, global_step: 590, interval_runtime: 0.9274, interval_samples_per_second: 8.626, interval_steps_per_second: 10.782, epoch: 29.5[0m
[32m[2022-08-31 15:40:02,947] [    INFO][0m - loss: 0.00225853, learning_rate: 1.2000000000000002e-06, global_step: 600, interval_runtime: 0.8901, interval_samples_per_second: 8.988, interval_steps_per_second: 11.235, epoch: 30.0[0m
[32m[2022-08-31 15:40:02,947] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 15:40:02,947] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 15:40:02,947] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:40:02,947] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:40:02,948] [    INFO][0m -   Total prediction steps = 3[0m
[32m[2022-08-31 15:40:03,630] [    INFO][0m - eval_loss: 1.3307230472564697, eval_accuracy: 0.775, eval_runtime: 0.6826, eval_samples_per_second: 234.387, eval_steps_per_second: 4.395, epoch: 30.0[0m
[32m[2022-08-31 15:40:03,631] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-600[0m
[32m[2022-08-31 15:40:03,631] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:40:06,968] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-600/tokenizer_config.json[0m
[32m[2022-08-31 15:40:06,969] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-600/special_tokens_map.json[0m
[32m[2022-08-31 15:40:13,954] [    INFO][0m - loss: 0.02700423, learning_rate: 1.17e-06, global_step: 610, interval_runtime: 11.0076, interval_samples_per_second: 0.727, interval_steps_per_second: 0.908, epoch: 30.5[0m
[32m[2022-08-31 15:40:14,831] [    INFO][0m - loss: 0.03048186, learning_rate: 1.14e-06, global_step: 620, interval_runtime: 0.8768, interval_samples_per_second: 9.124, interval_steps_per_second: 11.405, epoch: 31.0[0m
[32m[2022-08-31 15:40:15,759] [    INFO][0m - loss: 0.00384449, learning_rate: 1.11e-06, global_step: 630, interval_runtime: 0.9281, interval_samples_per_second: 8.62, interval_steps_per_second: 10.775, epoch: 31.5[0m
[32m[2022-08-31 15:40:16,639] [    INFO][0m - loss: 0.02724885, learning_rate: 1.08e-06, global_step: 640, interval_runtime: 0.8801, interval_samples_per_second: 9.09, interval_steps_per_second: 11.362, epoch: 32.0[0m
[32m[2022-08-31 15:40:17,577] [    INFO][0m - loss: 0.00534301, learning_rate: 1.05e-06, global_step: 650, interval_runtime: 0.9375, interval_samples_per_second: 8.533, interval_steps_per_second: 10.667, epoch: 32.5[0m
[32m[2022-08-31 15:40:18,454] [    INFO][0m - loss: 0.06061333, learning_rate: 1.0200000000000002e-06, global_step: 660, interval_runtime: 0.8775, interval_samples_per_second: 9.117, interval_steps_per_second: 11.396, epoch: 33.0[0m
[32m[2022-08-31 15:40:19,384] [    INFO][0m - loss: 0.01763411, learning_rate: 9.9e-07, global_step: 670, interval_runtime: 0.9292, interval_samples_per_second: 8.609, interval_steps_per_second: 10.762, epoch: 33.5[0m
[32m[2022-08-31 15:40:20,263] [    INFO][0m - loss: 0.07131995, learning_rate: 9.600000000000001e-07, global_step: 680, interval_runtime: 0.8792, interval_samples_per_second: 9.099, interval_steps_per_second: 11.374, epoch: 34.0[0m
[32m[2022-08-31 15:40:21,209] [    INFO][0m - loss: 0.00346733, learning_rate: 9.3e-07, global_step: 690, interval_runtime: 0.9457, interval_samples_per_second: 8.46, interval_steps_per_second: 10.575, epoch: 34.5[0m
[32m[2022-08-31 15:40:22,090] [    INFO][0m - loss: 0.07722953, learning_rate: 9e-07, global_step: 700, interval_runtime: 0.881, interval_samples_per_second: 9.081, interval_steps_per_second: 11.351, epoch: 35.0[0m
[32m[2022-08-31 15:40:22,090] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 15:40:22,091] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 15:40:22,091] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:40:22,091] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:40:22,091] [    INFO][0m -   Total prediction steps = 3[0m
[32m[2022-08-31 15:40:22,762] [    INFO][0m - eval_loss: 1.537071943283081, eval_accuracy: 0.79375, eval_runtime: 0.6713, eval_samples_per_second: 238.338, eval_steps_per_second: 4.469, epoch: 35.0[0m
[32m[2022-08-31 15:40:22,763] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-700[0m
[32m[2022-08-31 15:40:22,763] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:40:25,921] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-700/tokenizer_config.json[0m
[32m[2022-08-31 15:40:25,922] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-700/special_tokens_map.json[0m
[32m[2022-08-31 15:40:32,782] [    INFO][0m - loss: 0.00149372, learning_rate: 8.699999999999999e-07, global_step: 710, interval_runtime: 10.6927, interval_samples_per_second: 0.748, interval_steps_per_second: 0.935, epoch: 35.5[0m
[32m[2022-08-31 15:40:33,664] [    INFO][0m - loss: 0.00821707, learning_rate: 8.400000000000001e-07, global_step: 720, interval_runtime: 0.8823, interval_samples_per_second: 9.067, interval_steps_per_second: 11.334, epoch: 36.0[0m
[32m[2022-08-31 15:40:34,596] [    INFO][0m - loss: 0.00297164, learning_rate: 8.100000000000001e-07, global_step: 730, interval_runtime: 0.9317, interval_samples_per_second: 8.587, interval_steps_per_second: 10.733, epoch: 36.5[0m
[32m[2022-08-31 15:40:35,475] [    INFO][0m - loss: 0.00415947, learning_rate: 7.8e-07, global_step: 740, interval_runtime: 0.8791, interval_samples_per_second: 9.1, interval_steps_per_second: 11.375, epoch: 37.0[0m
[32m[2022-08-31 15:40:36,410] [    INFO][0m - loss: 0.01328269, learning_rate: 7.5e-07, global_step: 750, interval_runtime: 0.9344, interval_samples_per_second: 8.562, interval_steps_per_second: 10.702, epoch: 37.5[0m
[32m[2022-08-31 15:40:37,287] [    INFO][0m - loss: 0.09995848, learning_rate: 7.2e-07, global_step: 760, interval_runtime: 0.8776, interval_samples_per_second: 9.115, interval_steps_per_second: 11.394, epoch: 38.0[0m
[32m[2022-08-31 15:40:38,231] [    INFO][0m - loss: 0.00134489, learning_rate: 6.900000000000001e-07, global_step: 770, interval_runtime: 0.943, interval_samples_per_second: 8.483, interval_steps_per_second: 10.604, epoch: 38.5[0m
[32m[2022-08-31 15:40:39,120] [    INFO][0m - loss: 0.01542267, learning_rate: 6.6e-07, global_step: 780, interval_runtime: 0.8889, interval_samples_per_second: 9.0, interval_steps_per_second: 11.25, epoch: 39.0[0m
[32m[2022-08-31 15:40:40,050] [    INFO][0m - loss: 0.00251869, learning_rate: 6.3e-07, global_step: 790, interval_runtime: 0.9305, interval_samples_per_second: 8.598, interval_steps_per_second: 10.747, epoch: 39.5[0m
[32m[2022-08-31 15:40:40,930] [    INFO][0m - loss: 0.0318939, learning_rate: 6.000000000000001e-07, global_step: 800, interval_runtime: 0.8797, interval_samples_per_second: 9.094, interval_steps_per_second: 11.368, epoch: 40.0[0m
[32m[2022-08-31 15:40:40,930] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 15:40:40,931] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 15:40:40,931] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:40:40,931] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:40:40,931] [    INFO][0m -   Total prediction steps = 3[0m
[32m[2022-08-31 15:40:41,618] [    INFO][0m - eval_loss: 1.6436573266983032, eval_accuracy: 0.8, eval_runtime: 0.6867, eval_samples_per_second: 233.006, eval_steps_per_second: 4.369, epoch: 40.0[0m
[32m[2022-08-31 15:40:41,618] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-800[0m
[32m[2022-08-31 15:40:41,618] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:40:45,041] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-800/tokenizer_config.json[0m
[32m[2022-08-31 15:40:45,041] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-800/special_tokens_map.json[0m
[32m[2022-08-31 15:40:51,569] [    INFO][0m - loss: 0.00396234, learning_rate: 5.7e-07, global_step: 810, interval_runtime: 10.639, interval_samples_per_second: 0.752, interval_steps_per_second: 0.94, epoch: 40.5[0m
[32m[2022-08-31 15:40:52,453] [    INFO][0m - loss: 0.00189096, learning_rate: 5.4e-07, global_step: 820, interval_runtime: 0.8841, interval_samples_per_second: 9.049, interval_steps_per_second: 11.311, epoch: 41.0[0m
[32m[2022-08-31 15:40:53,390] [    INFO][0m - loss: 0.08729293, learning_rate: 5.100000000000001e-07, global_step: 830, interval_runtime: 0.9365, interval_samples_per_second: 8.543, interval_steps_per_second: 10.678, epoch: 41.5[0m
[32m[2022-08-31 15:40:54,276] [    INFO][0m - loss: 0.00548629, learning_rate: 4.800000000000001e-07, global_step: 840, interval_runtime: 0.8872, interval_samples_per_second: 9.017, interval_steps_per_second: 11.271, epoch: 42.0[0m
[32m[2022-08-31 15:40:55,214] [    INFO][0m - loss: 0.0092311, learning_rate: 4.5e-07, global_step: 850, interval_runtime: 0.9376, interval_samples_per_second: 8.533, interval_steps_per_second: 10.666, epoch: 42.5[0m
[32m[2022-08-31 15:40:56,095] [    INFO][0m - loss: 0.0007259, learning_rate: 4.2000000000000006e-07, global_step: 860, interval_runtime: 0.8808, interval_samples_per_second: 9.083, interval_steps_per_second: 11.353, epoch: 43.0[0m
[32m[2022-08-31 15:40:57,023] [    INFO][0m - loss: 0.09620624, learning_rate: 3.9e-07, global_step: 870, interval_runtime: 0.9287, interval_samples_per_second: 8.614, interval_steps_per_second: 10.768, epoch: 43.5[0m
[32m[2022-08-31 15:40:57,902] [    INFO][0m - loss: 0.05084218, learning_rate: 3.6e-07, global_step: 880, interval_runtime: 0.8788, interval_samples_per_second: 9.104, interval_steps_per_second: 11.38, epoch: 44.0[0m
[32m[2022-08-31 15:40:58,843] [    INFO][0m - loss: 0.02862762, learning_rate: 3.3e-07, global_step: 890, interval_runtime: 0.9404, interval_samples_per_second: 8.507, interval_steps_per_second: 10.633, epoch: 44.5[0m
[32m[2022-08-31 15:40:59,723] [    INFO][0m - loss: 0.02995912, learning_rate: 3.0000000000000004e-07, global_step: 900, interval_runtime: 0.8801, interval_samples_per_second: 9.09, interval_steps_per_second: 11.362, epoch: 45.0[0m
[32m[2022-08-31 15:40:59,723] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 15:40:59,723] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 15:40:59,724] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:40:59,724] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:40:59,724] [    INFO][0m -   Total prediction steps = 3[0m
[32m[2022-08-31 15:41:00,389] [    INFO][0m - eval_loss: 1.6883466243743896, eval_accuracy: 0.79375, eval_runtime: 0.665, eval_samples_per_second: 240.586, eval_steps_per_second: 4.511, epoch: 45.0[0m
[32m[2022-08-31 15:41:00,389] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-900[0m
[32m[2022-08-31 15:41:00,389] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:41:03,618] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-900/tokenizer_config.json[0m
[32m[2022-08-31 15:41:03,618] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-900/special_tokens_map.json[0m
[32m[2022-08-31 15:41:09,881] [    INFO][0m - loss: 0.03727016, learning_rate: 2.7e-07, global_step: 910, interval_runtime: 10.158, interval_samples_per_second: 0.788, interval_steps_per_second: 0.984, epoch: 45.5[0m
[32m[2022-08-31 15:41:10,764] [    INFO][0m - loss: 0.1114477, learning_rate: 2.4000000000000003e-07, global_step: 920, interval_runtime: 0.8829, interval_samples_per_second: 9.061, interval_steps_per_second: 11.326, epoch: 46.0[0m
[32m[2022-08-31 15:41:11,701] [    INFO][0m - loss: 0.00055941, learning_rate: 2.1000000000000003e-07, global_step: 930, interval_runtime: 0.9374, interval_samples_per_second: 8.534, interval_steps_per_second: 10.668, epoch: 46.5[0m
[32m[2022-08-31 15:41:12,579] [    INFO][0m - loss: 0.04395221, learning_rate: 1.8e-07, global_step: 940, interval_runtime: 0.8771, interval_samples_per_second: 9.121, interval_steps_per_second: 11.401, epoch: 47.0[0m
[32m[2022-08-31 15:41:13,573] [    INFO][0m - loss: 0.00444442, learning_rate: 1.5000000000000002e-07, global_step: 950, interval_runtime: 0.9948, interval_samples_per_second: 8.042, interval_steps_per_second: 10.052, epoch: 47.5[0m
[32m[2022-08-31 15:41:14,459] [    INFO][0m - loss: 0.0271625, learning_rate: 1.2000000000000002e-07, global_step: 960, interval_runtime: 0.8854, interval_samples_per_second: 9.035, interval_steps_per_second: 11.294, epoch: 48.0[0m
[32m[2022-08-31 15:41:15,424] [    INFO][0m - loss: 0.08527821, learning_rate: 9e-08, global_step: 970, interval_runtime: 0.9655, interval_samples_per_second: 8.286, interval_steps_per_second: 10.357, epoch: 48.5[0m
[32m[2022-08-31 15:41:16,304] [    INFO][0m - loss: 0.02967793, learning_rate: 6.000000000000001e-08, global_step: 980, interval_runtime: 0.8801, interval_samples_per_second: 9.09, interval_steps_per_second: 11.363, epoch: 49.0[0m
[32m[2022-08-31 15:41:17,234] [    INFO][0m - loss: 0.00143402, learning_rate: 3.0000000000000004e-08, global_step: 990, interval_runtime: 0.9297, interval_samples_per_second: 8.605, interval_steps_per_second: 10.756, epoch: 49.5[0m
[32m[2022-08-31 15:41:18,173] [    INFO][0m - loss: 0.02356675, learning_rate: 0.0, global_step: 1000, interval_runtime: 0.9387, interval_samples_per_second: 8.522, interval_steps_per_second: 10.653, epoch: 50.0[0m
[32m[2022-08-31 15:41:18,173] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 15:41:18,173] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 15:41:18,173] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:41:18,174] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:41:18,174] [    INFO][0m -   Total prediction steps = 3[0m
[32m[2022-08-31 15:41:18,853] [    INFO][0m - eval_loss: 1.700993299484253, eval_accuracy: 0.79375, eval_runtime: 0.6792, eval_samples_per_second: 235.583, eval_steps_per_second: 4.417, epoch: 50.0[0m
[32m[2022-08-31 15:41:18,853] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-1000[0m
[32m[2022-08-31 15:41:18,854] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:41:22,460] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-1000/tokenizer_config.json[0m
[32m[2022-08-31 15:41:22,461] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-1000/special_tokens_map.json[0m
[32m[2022-08-31 15:41:28,020] [    INFO][0m - 
Training completed. 
[0m
[32m[2022-08-31 15:41:28,020] [    INFO][0m - Loading best model from ./checkpoints/checkpoint-800 (score: 0.8).[0m
[32m[2022-08-31 15:41:28,994] [    INFO][0m - train_runtime: 188.0339, train_samples_per_second: 42.546, train_steps_per_second: 5.318, train_loss: 0.15226376488618554, epoch: 50.0[0m
[32m[2022-08-31 15:41:29,040] [    INFO][0m - Saving model checkpoint to ./checkpoints/[0m
[32m[2022-08-31 15:41:29,040] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:41:32,461] [    INFO][0m - tokenizer config file saved in ./checkpoints/tokenizer_config.json[0m
[32m[2022-08-31 15:41:32,462] [    INFO][0m - Special tokens file saved in ./checkpoints/special_tokens_map.json[0m
[32m[2022-08-31 15:41:32,464] [    INFO][0m - ***** train metrics *****[0m
[32m[2022-08-31 15:41:32,464] [    INFO][0m -   epoch                    =       50.0[0m
[32m[2022-08-31 15:41:32,465] [    INFO][0m -   train_loss               =     0.1523[0m
[32m[2022-08-31 15:41:32,465] [    INFO][0m -   train_runtime            = 0:03:08.03[0m
[32m[2022-08-31 15:41:32,465] [    INFO][0m -   train_samples_per_second =     42.546[0m
[32m[2022-08-31 15:41:32,465] [    INFO][0m -   train_steps_per_second   =      5.318[0m
[32m[2022-08-31 15:41:32,470] [    INFO][0m - ***** Running Prediction *****[0m
[32m[2022-08-31 15:41:32,471] [    INFO][0m -   Num examples = 1772[0m
[32m[2022-08-31 15:41:32,471] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:41:32,471] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:41:32,471] [    INFO][0m -   Total prediction steps = 28[0m
[32m[2022-08-31 15:41:39,917] [    INFO][0m - ***** test metrics *****[0m
[32m[2022-08-31 15:41:39,918] [    INFO][0m -   test_accuracy           =     0.7466[0m
[32m[2022-08-31 15:41:39,918] [    INFO][0m -   test_loss               =     1.9669[0m
[32m[2022-08-31 15:41:39,918] [    INFO][0m -   test_runtime            = 0:00:07.44[0m
[32m[2022-08-31 15:41:39,918] [    INFO][0m -   test_samples_per_second =    237.953[0m
[32m[2022-08-31 15:41:39,918] [    INFO][0m -   test_steps_per_second   =       3.76[0m
[32m[2022-08-31 15:41:39,919] [    INFO][0m - ***** Running Prediction *****[0m
[32m[2022-08-31 15:41:39,919] [    INFO][0m -   Num examples = 2000[0m
[32m[2022-08-31 15:41:39,919] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:41:39,919] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:41:39,919] [    INFO][0m -   Total prediction steps = 32[0m
[32m[2022-08-31 15:41:49,982] [    INFO][0m - Predictions for bustm saved to ./fewclue_submit_examples.[0m
[]
run.sh: line 72: --freeze_plm: command not found
