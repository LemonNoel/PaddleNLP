[33m[2022-08-31 17:07:49,598] [ WARNING][0m - evaluation_strategy reset to IntervalStrategy.STEPS for do_eval is True. you can also set evaluation_strategy='epoch'.[0m
[32m[2022-08-31 17:07:49,598] [    INFO][0m - The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).[0m
[32m[2022-08-31 17:07:49,599] [    INFO][0m - ============================================================[0m
[32m[2022-08-31 17:07:49,599] [    INFO][0m -      Model Configuration Arguments      [0m
[32m[2022-08-31 17:07:49,599] [    INFO][0m - paddle commit id              :65f388690048d0965ec7d3b43fb6ee9d8c6dee7c[0m
[32m[2022-08-31 17:07:49,599] [    INFO][0m - do_save                       :True[0m
[32m[2022-08-31 17:07:49,599] [    INFO][0m - do_test                       :True[0m
[32m[2022-08-31 17:07:49,599] [    INFO][0m - early_stop_patience           :4[0m
[32m[2022-08-31 17:07:49,599] [    INFO][0m - export_type                   :paddle[0m
[32m[2022-08-31 17:07:49,599] [    INFO][0m - model_name_or_path            :roformer_v2_chinese_char_base[0m
[32m[2022-08-31 17:07:49,599] [    INFO][0m - [0m
[32m[2022-08-31 17:07:49,599] [    INFO][0m - ============================================================[0m
[32m[2022-08-31 17:07:49,600] [    INFO][0m -       Data Configuration Arguments      [0m
[32m[2022-08-31 17:07:49,600] [    INFO][0m - paddle commit id              :65f388690048d0965ec7d3b43fb6ee9d8c6dee7c[0m
[32m[2022-08-31 17:07:49,600] [    INFO][0m - encoder_hidden_size           :200[0m
[32m[2022-08-31 17:07:49,600] [    INFO][0m - prompt                        :{'text':'text_a'}{'soft':'这款应用属于'}{'mask'}{'mask'}{'soft':'类别.'}[0m
[32m[2022-08-31 17:07:49,600] [    INFO][0m - soft_encoder                  :lstm[0m
[32m[2022-08-31 17:07:49,600] [    INFO][0m - split_id                      :few_all[0m
[32m[2022-08-31 17:07:49,600] [    INFO][0m - task_name                     :iflytek[0m
[32m[2022-08-31 17:07:49,600] [    INFO][0m - [0m
[32m[2022-08-31 17:07:49,600] [    INFO][0m - Already cached /ssd2/wanghuijuan03/.paddlenlp/models/roformer_v2_chinese_char_base/model_state.pdparams[0m
W0831 17:07:49.602249 79435 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 11.2
W0831 17:07:49.606330 79435 gpu_resources.cc:91] device: 0, cuDNN Version: 8.1.
[32m[2022-08-31 17:07:52,432] [    INFO][0m - Already cached /ssd2/wanghuijuan03/.paddlenlp/models/roformer_v2_chinese_char_base/vocab.txt[0m
[32m[2022-08-31 17:07:52,440] [    INFO][0m - tokenizer config file saved in /ssd2/wanghuijuan03/.paddlenlp/models/roformer_v2_chinese_char_base/tokenizer_config.json[0m
[32m[2022-08-31 17:07:52,440] [    INFO][0m - Special tokens file saved in /ssd2/wanghuijuan03/.paddlenlp/models/roformer_v2_chinese_char_base/special_tokens_map.json[0m
[33m[2022-08-31 17:07:52,447] [ WARNING][0m - Encoder has already set as lstm, change `prompt_encoder` will reset parameters.[0m
[32m[2022-08-31 17:07:52,467] [    INFO][0m - Using template: [{'add_prefix_space': '', 'text': 'text_a'}, {'add_prefix_space': '', 'soft': '这'}, {'add_prefix_space': '', 'soft': '款'}, {'add_prefix_space': '', 'soft': '应'}, {'add_prefix_space': '', 'soft': '用'}, {'add_prefix_space': '', 'soft': '属'}, {'add_prefix_space': '', 'soft': '于'}, {'add_prefix_space': '', 'mask': None}, {'add_prefix_space': '', 'mask': None}, {'add_prefix_space': '', 'soft': '类'}, {'add_prefix_space': '', 'soft': '别'}, {'add_prefix_space': '', 'soft': '.'}][0m
2022-08-31 17:07:52,468 INFO [download.py:119] unique_endpoints {''}
[32m[2022-08-31 17:07:52,645] [    INFO][0m - ============================================================[0m
[32m[2022-08-31 17:07:52,645] [    INFO][0m -     Training Configuration Arguments    [0m
[32m[2022-08-31 17:07:52,645] [    INFO][0m - paddle commit id              :65f388690048d0965ec7d3b43fb6ee9d8c6dee7c[0m
[32m[2022-08-31 17:07:52,645] [    INFO][0m - _no_sync_in_gradient_accumulation:True[0m
[32m[2022-08-31 17:07:52,645] [    INFO][0m - adam_beta1                    :0.9[0m
[32m[2022-08-31 17:07:52,645] [    INFO][0m - adam_beta2                    :0.999[0m
[32m[2022-08-31 17:07:52,645] [    INFO][0m - adam_epsilon                  :1e-08[0m
[32m[2022-08-31 17:07:52,645] [    INFO][0m - alpha_rdrop                   :5.0[0m
[32m[2022-08-31 17:07:52,646] [    INFO][0m - alpha_rgl                     :0.5[0m
[32m[2022-08-31 17:07:52,646] [    INFO][0m - current_device                :gpu:0[0m
[32m[2022-08-31 17:07:52,646] [    INFO][0m - dataloader_drop_last          :False[0m
[32m[2022-08-31 17:07:52,646] [    INFO][0m - dataloader_num_workers        :0[0m
[32m[2022-08-31 17:07:52,646] [    INFO][0m - device                        :gpu[0m
[32m[2022-08-31 17:07:52,646] [    INFO][0m - disable_tqdm                  :True[0m
[32m[2022-08-31 17:07:52,646] [    INFO][0m - do_eval                       :True[0m
[32m[2022-08-31 17:07:52,646] [    INFO][0m - do_export                     :False[0m
[32m[2022-08-31 17:07:52,646] [    INFO][0m - do_predict                    :True[0m
[32m[2022-08-31 17:07:52,646] [    INFO][0m - do_train                      :True[0m
[32m[2022-08-31 17:07:52,646] [    INFO][0m - eval_batch_size               :32[0m
[32m[2022-08-31 17:07:52,646] [    INFO][0m - eval_steps                    :100[0m
[32m[2022-08-31 17:07:52,647] [    INFO][0m - evaluation_strategy           :IntervalStrategy.STEPS[0m
[32m[2022-08-31 17:07:52,647] [    INFO][0m - first_max_length              :None[0m
[32m[2022-08-31 17:07:52,647] [    INFO][0m - fp16                          :False[0m
[32m[2022-08-31 17:07:52,647] [    INFO][0m - fp16_opt_level                :O1[0m
[32m[2022-08-31 17:07:52,647] [    INFO][0m - freeze_dropout                :False[0m
[32m[2022-08-31 17:07:52,647] [    INFO][0m - freeze_plm                    :False[0m
[32m[2022-08-31 17:07:52,647] [    INFO][0m - gradient_accumulation_steps   :1[0m
[32m[2022-08-31 17:07:52,647] [    INFO][0m - greater_is_better             :True[0m
[32m[2022-08-31 17:07:52,647] [    INFO][0m - ignore_data_skip              :False[0m
[32m[2022-08-31 17:07:52,647] [    INFO][0m - label_names                   :None[0m
[32m[2022-08-31 17:07:52,647] [    INFO][0m - learning_rate                 :3e-05[0m
[32m[2022-08-31 17:07:52,647] [    INFO][0m - load_best_model_at_end        :True[0m
[32m[2022-08-31 17:07:52,647] [    INFO][0m - local_process_index           :0[0m
[32m[2022-08-31 17:07:52,647] [    INFO][0m - local_rank                    :-1[0m
[32m[2022-08-31 17:07:52,648] [    INFO][0m - log_level                     :-1[0m
[32m[2022-08-31 17:07:52,648] [    INFO][0m - log_level_replica             :-1[0m
[32m[2022-08-31 17:07:52,648] [    INFO][0m - log_on_each_node              :True[0m
[32m[2022-08-31 17:07:52,648] [    INFO][0m - logging_dir                   :./checkpoints/runs/Aug31_17-07-49_instance-3bwob41y-01[0m
[32m[2022-08-31 17:07:52,648] [    INFO][0m - logging_first_step            :False[0m
[32m[2022-08-31 17:07:52,648] [    INFO][0m - logging_steps                 :10[0m
[32m[2022-08-31 17:07:52,648] [    INFO][0m - logging_strategy              :IntervalStrategy.STEPS[0m
[32m[2022-08-31 17:07:52,648] [    INFO][0m - lr_scheduler_type             :SchedulerType.LINEAR[0m
[32m[2022-08-31 17:07:52,648] [    INFO][0m - max_grad_norm                 :1.0[0m
[32m[2022-08-31 17:07:52,648] [    INFO][0m - max_seq_length                :320[0m
[32m[2022-08-31 17:07:52,648] [    INFO][0m - max_steps                     :-1[0m
[32m[2022-08-31 17:07:52,648] [    INFO][0m - metric_for_best_model         :accuracy[0m
[32m[2022-08-31 17:07:52,648] [    INFO][0m - minimum_eval_times            :None[0m
[32m[2022-08-31 17:07:52,648] [    INFO][0m - no_cuda                       :False[0m
[32m[2022-08-31 17:07:52,649] [    INFO][0m - num_train_epochs              :20.0[0m
[32m[2022-08-31 17:07:52,649] [    INFO][0m - optim                         :OptimizerNames.ADAMW[0m
[32m[2022-08-31 17:07:52,649] [    INFO][0m - other_max_length              :None[0m
[32m[2022-08-31 17:07:52,649] [    INFO][0m - output_dir                    :./checkpoints/[0m
[32m[2022-08-31 17:07:52,649] [    INFO][0m - overwrite_output_dir          :False[0m
[32m[2022-08-31 17:07:52,649] [    INFO][0m - past_index                    :-1[0m
[32m[2022-08-31 17:07:52,649] [    INFO][0m - per_device_eval_batch_size    :32[0m
[32m[2022-08-31 17:07:52,649] [    INFO][0m - per_device_train_batch_size   :8[0m
[32m[2022-08-31 17:07:52,649] [    INFO][0m - ppt_adam_beta1                :0.9[0m
[32m[2022-08-31 17:07:52,649] [    INFO][0m - ppt_adam_beta2                :0.999[0m
[32m[2022-08-31 17:07:52,649] [    INFO][0m - ppt_adam_epsilon              :1e-08[0m
[32m[2022-08-31 17:07:52,649] [    INFO][0m - ppt_learning_rate             :0.0003[0m
[32m[2022-08-31 17:07:52,649] [    INFO][0m - ppt_weight_decay              :0.0[0m
[32m[2022-08-31 17:07:52,649] [    INFO][0m - prediction_loss_only          :False[0m
[32m[2022-08-31 17:07:52,650] [    INFO][0m - process_index                 :0[0m
[32m[2022-08-31 17:07:52,650] [    INFO][0m - remove_unused_columns         :True[0m
[32m[2022-08-31 17:07:52,650] [    INFO][0m - report_to                     :['visualdl'][0m
[32m[2022-08-31 17:07:52,650] [    INFO][0m - resume_from_checkpoint        :None[0m
[32m[2022-08-31 17:07:52,650] [    INFO][0m - run_name                      :./checkpoints/[0m
[32m[2022-08-31 17:07:52,650] [    INFO][0m - save_on_each_node             :False[0m
[32m[2022-08-31 17:07:52,650] [    INFO][0m - save_steps                    :100[0m
[32m[2022-08-31 17:07:52,650] [    INFO][0m - save_strategy                 :IntervalStrategy.STEPS[0m
[32m[2022-08-31 17:07:52,650] [    INFO][0m - save_total_limit              :None[0m
[32m[2022-08-31 17:07:52,650] [    INFO][0m - scale_loss                    :32768[0m
[32m[2022-08-31 17:07:52,650] [    INFO][0m - seed                          :42[0m
[32m[2022-08-31 17:07:52,650] [    INFO][0m - should_log                    :True[0m
[32m[2022-08-31 17:07:52,650] [    INFO][0m - should_save                   :True[0m
[32m[2022-08-31 17:07:52,650] [    INFO][0m - task_type                     :multi-class[0m
[32m[2022-08-31 17:07:52,651] [    INFO][0m - train_batch_size              :8[0m
[32m[2022-08-31 17:07:52,651] [    INFO][0m - truncate_mode                 :tail[0m
[32m[2022-08-31 17:07:52,651] [    INFO][0m - use_rdrop                     :False[0m
[32m[2022-08-31 17:07:52,651] [    INFO][0m - use_rgl                       :False[0m
[32m[2022-08-31 17:07:52,651] [    INFO][0m - warmup_ratio                  :0.0[0m
[32m[2022-08-31 17:07:52,651] [    INFO][0m - warmup_steps                  :0[0m
[32m[2022-08-31 17:07:52,651] [    INFO][0m - weight_decay                  :0.0[0m
[32m[2022-08-31 17:07:52,651] [    INFO][0m - world_size                    :1[0m
[32m[2022-08-31 17:07:52,651] [    INFO][0m - [0m
[32m[2022-08-31 17:07:52,653] [    INFO][0m - ***** Running training *****[0m
[32m[2022-08-31 17:07:52,653] [    INFO][0m -   Num examples = 3024[0m
[32m[2022-08-31 17:07:52,653] [    INFO][0m -   Num Epochs = 20[0m
[32m[2022-08-31 17:07:52,653] [    INFO][0m -   Instantaneous batch size per device = 8[0m
[32m[2022-08-31 17:07:52,653] [    INFO][0m -   Total train batch size (w. parallel, distributed & accumulation) = 8[0m
[32m[2022-08-31 17:07:52,653] [    INFO][0m -   Gradient Accumulation steps = 1[0m
[32m[2022-08-31 17:07:52,653] [    INFO][0m -   Total optimization steps = 7560.0[0m
[32m[2022-08-31 17:07:52,653] [    INFO][0m -   Total num train samples = 60480[0m
[32m[2022-08-31 17:07:55,738] [    INFO][0m - loss: 5.408041, learning_rate: 2.996031746031746e-05, global_step: 10, interval_runtime: 3.084, interval_samples_per_second: 2.594, interval_steps_per_second: 3.243, epoch: 0.0265[0m
[32m[2022-08-31 17:07:57,853] [    INFO][0m - loss: 5.41471977, learning_rate: 2.992063492063492e-05, global_step: 20, interval_runtime: 2.1151, interval_samples_per_second: 3.782, interval_steps_per_second: 4.728, epoch: 0.0529[0m
[32m[2022-08-31 17:08:00,000] [    INFO][0m - loss: 5.22729721, learning_rate: 2.9880952380952383e-05, global_step: 30, interval_runtime: 2.1464, interval_samples_per_second: 3.727, interval_steps_per_second: 4.659, epoch: 0.0794[0m
[32m[2022-08-31 17:08:02,173] [    INFO][0m - loss: 5.07568436, learning_rate: 2.984126984126984e-05, global_step: 40, interval_runtime: 2.1735, interval_samples_per_second: 3.681, interval_steps_per_second: 4.601, epoch: 0.1058[0m
[32m[2022-08-31 17:08:04,417] [    INFO][0m - loss: 5.07851486, learning_rate: 2.98015873015873e-05, global_step: 50, interval_runtime: 2.244, interval_samples_per_second: 3.565, interval_steps_per_second: 4.456, epoch: 0.1323[0m
[32m[2022-08-31 17:08:06,655] [    INFO][0m - loss: 5.03366585, learning_rate: 2.9761904761904762e-05, global_step: 60, interval_runtime: 2.2372, interval_samples_per_second: 3.576, interval_steps_per_second: 4.47, epoch: 0.1587[0m
[32m[2022-08-31 17:08:08,887] [    INFO][0m - loss: 5.23741264, learning_rate: 2.9722222222222223e-05, global_step: 70, interval_runtime: 2.2323, interval_samples_per_second: 3.584, interval_steps_per_second: 4.48, epoch: 0.1852[0m
[32m[2022-08-31 17:08:11,134] [    INFO][0m - loss: 4.76879578, learning_rate: 2.9682539682539683e-05, global_step: 80, interval_runtime: 2.2471, interval_samples_per_second: 3.56, interval_steps_per_second: 4.45, epoch: 0.2116[0m
[32m[2022-08-31 17:08:13,374] [    INFO][0m - loss: 4.78171387, learning_rate: 2.9642857142857144e-05, global_step: 90, interval_runtime: 2.2397, interval_samples_per_second: 3.572, interval_steps_per_second: 4.465, epoch: 0.2381[0m
[32m[2022-08-31 17:08:15,648] [    INFO][0m - loss: 4.43004913, learning_rate: 2.96031746031746e-05, global_step: 100, interval_runtime: 2.274, interval_samples_per_second: 3.518, interval_steps_per_second: 4.398, epoch: 0.2646[0m
[32m[2022-08-31 17:08:15,649] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 17:08:15,649] [    INFO][0m -   Num examples = 1373[0m
[32m[2022-08-31 17:08:15,649] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 17:08:15,649] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 17:08:15,649] [    INFO][0m -   Total prediction steps = 43[0m
[32m[2022-08-31 17:08:32,571] [    INFO][0m - eval_loss: 3.838409423828125, eval_accuracy: 0.17407137654770574, eval_runtime: 16.9209, eval_samples_per_second: 81.142, eval_steps_per_second: 2.541, epoch: 0.2646[0m
[32m[2022-08-31 17:08:32,571] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-100[0m
[32m[2022-08-31 17:08:32,572] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 17:08:33,678] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-100/tokenizer_config.json[0m
[32m[2022-08-31 17:08:33,678] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-100/special_tokens_map.json[0m
[32m[2022-08-31 17:08:37,700] [    INFO][0m - loss: 4.05166168, learning_rate: 2.9563492063492066e-05, global_step: 110, interval_runtime: 22.052, interval_samples_per_second: 0.363, interval_steps_per_second: 0.453, epoch: 0.291[0m
[32m[2022-08-31 17:08:40,090] [    INFO][0m - loss: 3.77155991, learning_rate: 2.9523809523809523e-05, global_step: 120, interval_runtime: 2.3899, interval_samples_per_second: 3.347, interval_steps_per_second: 4.184, epoch: 0.3175[0m
[32m[2022-08-31 17:08:42,479] [    INFO][0m - loss: 3.82090797, learning_rate: 2.9484126984126984e-05, global_step: 130, interval_runtime: 2.389, interval_samples_per_second: 3.349, interval_steps_per_second: 4.186, epoch: 0.3439[0m
[32m[2022-08-31 17:08:45,656] [    INFO][0m - loss: 3.55142822, learning_rate: 2.9444444444444445e-05, global_step: 140, interval_runtime: 3.1771, interval_samples_per_second: 2.518, interval_steps_per_second: 3.148, epoch: 0.3704[0m
[32m[2022-08-31 17:08:48,097] [    INFO][0m - loss: 3.31094475, learning_rate: 2.9404761904761905e-05, global_step: 150, interval_runtime: 2.4407, interval_samples_per_second: 3.278, interval_steps_per_second: 4.097, epoch: 0.3968[0m
[32m[2022-08-31 17:08:50,531] [    INFO][0m - loss: 3.37072983, learning_rate: 2.9365079365079366e-05, global_step: 160, interval_runtime: 2.4347, interval_samples_per_second: 3.286, interval_steps_per_second: 4.107, epoch: 0.4233[0m
[32m[2022-08-31 17:08:53,077] [    INFO][0m - loss: 2.98227444, learning_rate: 2.9325396825396827e-05, global_step: 170, interval_runtime: 2.5453, interval_samples_per_second: 3.143, interval_steps_per_second: 3.929, epoch: 0.4497[0m
[32m[2022-08-31 17:08:55,571] [    INFO][0m - loss: 3.12673492, learning_rate: 2.9285714285714284e-05, global_step: 180, interval_runtime: 2.4941, interval_samples_per_second: 3.208, interval_steps_per_second: 4.009, epoch: 0.4762[0m
[32m[2022-08-31 17:08:58,125] [    INFO][0m - loss: 2.9036459, learning_rate: 2.9246031746031748e-05, global_step: 190, interval_runtime: 2.5537, interval_samples_per_second: 3.133, interval_steps_per_second: 3.916, epoch: 0.5026[0m
[32m[2022-08-31 17:09:00,724] [    INFO][0m - loss: 2.7742466, learning_rate: 2.9206349206349206e-05, global_step: 200, interval_runtime: 2.6, interval_samples_per_second: 3.077, interval_steps_per_second: 3.846, epoch: 0.5291[0m
[32m[2022-08-31 17:09:00,725] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 17:09:00,725] [    INFO][0m -   Num examples = 1373[0m
[32m[2022-08-31 17:09:00,725] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 17:09:00,725] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 17:09:00,725] [    INFO][0m -   Total prediction steps = 43[0m
[32m[2022-08-31 17:09:22,734] [    INFO][0m - eval_loss: 2.5237109661102295, eval_accuracy: 0.3721777130371449, eval_runtime: 22.0044, eval_samples_per_second: 62.396, eval_steps_per_second: 1.954, epoch: 0.5291[0m
[32m[2022-08-31 17:09:22,735] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-200[0m
[32m[2022-08-31 17:09:22,735] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 17:09:23,833] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-200/tokenizer_config.json[0m
[32m[2022-08-31 17:09:23,833] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-200/special_tokens_map.json[0m
[32m[2022-08-31 17:09:28,288] [    INFO][0m - loss: 2.77450333, learning_rate: 2.9166666666666666e-05, global_step: 210, interval_runtime: 27.5632, interval_samples_per_second: 0.29, interval_steps_per_second: 0.363, epoch: 0.5556[0m
[32m[2022-08-31 17:09:31,051] [    INFO][0m - loss: 2.65551682, learning_rate: 2.9126984126984127e-05, global_step: 220, interval_runtime: 2.7638, interval_samples_per_second: 2.895, interval_steps_per_second: 3.618, epoch: 0.582[0m
[32m[2022-08-31 17:09:33,827] [    INFO][0m - loss: 2.37265282, learning_rate: 2.9087301587301588e-05, global_step: 230, interval_runtime: 2.7755, interval_samples_per_second: 2.882, interval_steps_per_second: 3.603, epoch: 0.6085[0m
[32m[2022-08-31 17:09:36,664] [    INFO][0m - loss: 2.50203609, learning_rate: 2.904761904761905e-05, global_step: 240, interval_runtime: 2.8366, interval_samples_per_second: 2.82, interval_steps_per_second: 3.525, epoch: 0.6349[0m
[32m[2022-08-31 17:09:39,483] [    INFO][0m - loss: 2.62729893, learning_rate: 2.900793650793651e-05, global_step: 250, interval_runtime: 2.8197, interval_samples_per_second: 2.837, interval_steps_per_second: 3.546, epoch: 0.6614[0m
[32m[2022-08-31 17:09:42,427] [    INFO][0m - loss: 2.46006889, learning_rate: 2.8968253968253967e-05, global_step: 260, interval_runtime: 2.9434, interval_samples_per_second: 2.718, interval_steps_per_second: 3.397, epoch: 0.6878[0m
[32m[2022-08-31 17:09:45,314] [    INFO][0m - loss: 1.9626421, learning_rate: 2.892857142857143e-05, global_step: 270, interval_runtime: 2.8868, interval_samples_per_second: 2.771, interval_steps_per_second: 3.464, epoch: 0.7143[0m
[32m[2022-08-31 17:09:48,178] [    INFO][0m - loss: 2.61910877, learning_rate: 2.8888888888888888e-05, global_step: 280, interval_runtime: 2.865, interval_samples_per_second: 2.792, interval_steps_per_second: 3.49, epoch: 0.7407[0m
[32m[2022-08-31 17:09:51,155] [    INFO][0m - loss: 2.29571514, learning_rate: 2.884920634920635e-05, global_step: 290, interval_runtime: 2.9767, interval_samples_per_second: 2.688, interval_steps_per_second: 3.359, epoch: 0.7672[0m
[32m[2022-08-31 17:09:54,148] [    INFO][0m - loss: 2.25582943, learning_rate: 2.880952380952381e-05, global_step: 300, interval_runtime: 2.993, interval_samples_per_second: 2.673, interval_steps_per_second: 3.341, epoch: 0.7937[0m
[32m[2022-08-31 17:09:54,149] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 17:09:54,149] [    INFO][0m -   Num examples = 1373[0m
[32m[2022-08-31 17:09:54,149] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 17:09:54,149] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 17:09:54,149] [    INFO][0m -   Total prediction steps = 43[0m
[32m[2022-08-31 17:10:21,864] [    INFO][0m - eval_loss: 2.1769609451293945, eval_accuracy: 0.4748725418790969, eval_runtime: 27.7137, eval_samples_per_second: 49.542, eval_steps_per_second: 1.552, epoch: 0.7937[0m
[32m[2022-08-31 17:10:21,864] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-300[0m
[32m[2022-08-31 17:10:21,864] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 17:10:22,953] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-300/tokenizer_config.json[0m
[32m[2022-08-31 17:10:22,953] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-300/special_tokens_map.json[0m
[32m[2022-08-31 17:10:28,047] [    INFO][0m - loss: 2.82368164, learning_rate: 2.876984126984127e-05, global_step: 310, interval_runtime: 33.8987, interval_samples_per_second: 0.236, interval_steps_per_second: 0.295, epoch: 0.8201[0m
[32m[2022-08-31 17:10:31,159] [    INFO][0m - loss: 1.93388348, learning_rate: 2.873015873015873e-05, global_step: 320, interval_runtime: 3.1118, interval_samples_per_second: 2.571, interval_steps_per_second: 3.214, epoch: 0.8466[0m
[32m[2022-08-31 17:10:34,285] [    INFO][0m - loss: 2.47874928, learning_rate: 2.8690476190476192e-05, global_step: 330, interval_runtime: 3.1262, interval_samples_per_second: 2.559, interval_steps_per_second: 3.199, epoch: 0.873[0m
[32m[2022-08-31 17:10:37,423] [    INFO][0m - loss: 2.39831657, learning_rate: 2.865079365079365e-05, global_step: 340, interval_runtime: 3.138, interval_samples_per_second: 2.549, interval_steps_per_second: 3.187, epoch: 0.8995[0m
[32m[2022-08-31 17:10:40,597] [    INFO][0m - loss: 2.33720341, learning_rate: 2.8611111111111113e-05, global_step: 350, interval_runtime: 3.1743, interval_samples_per_second: 2.52, interval_steps_per_second: 3.15, epoch: 0.9259[0m
[32m[2022-08-31 17:10:43,780] [    INFO][0m - loss: 2.58483696, learning_rate: 2.857142857142857e-05, global_step: 360, interval_runtime: 3.1826, interval_samples_per_second: 2.514, interval_steps_per_second: 3.142, epoch: 0.9524[0m
[32m[2022-08-31 17:10:47,037] [    INFO][0m - loss: 2.34044113, learning_rate: 2.853174603174603e-05, global_step: 370, interval_runtime: 3.2572, interval_samples_per_second: 2.456, interval_steps_per_second: 3.07, epoch: 0.9788[0m
[32m[2022-08-31 17:10:50,489] [    INFO][0m - loss: 1.81886978, learning_rate: 2.8492063492063492e-05, global_step: 380, interval_runtime: 3.4522, interval_samples_per_second: 2.317, interval_steps_per_second: 2.897, epoch: 1.0053[0m
[32m[2022-08-31 17:10:53,773] [    INFO][0m - loss: 1.53623962, learning_rate: 2.8452380952380953e-05, global_step: 390, interval_runtime: 3.284, interval_samples_per_second: 2.436, interval_steps_per_second: 3.045, epoch: 1.0317[0m
[32m[2022-08-31 17:10:57,121] [    INFO][0m - loss: 1.66454296, learning_rate: 2.8412698412698414e-05, global_step: 400, interval_runtime: 3.3474, interval_samples_per_second: 2.39, interval_steps_per_second: 2.987, epoch: 1.0582[0m
[32m[2022-08-31 17:10:57,122] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 17:10:57,122] [    INFO][0m -   Num examples = 1373[0m
[32m[2022-08-31 17:10:57,122] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 17:10:57,122] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 17:10:57,122] [    INFO][0m -   Total prediction steps = 43[0m
[32m[2022-08-31 17:11:29,292] [    INFO][0m - eval_loss: 2.0952889919281006, eval_accuracy: 0.48725418790968683, eval_runtime: 32.1694, eval_samples_per_second: 42.68, eval_steps_per_second: 1.337, epoch: 1.0582[0m
[32m[2022-08-31 17:11:29,293] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-400[0m
[32m[2022-08-31 17:11:29,293] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 17:11:30,454] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-400/tokenizer_config.json[0m
[32m[2022-08-31 17:11:30,454] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-400/special_tokens_map.json[0m
[32m[2022-08-31 17:11:35,769] [    INFO][0m - loss: 1.67010117, learning_rate: 2.8373015873015875e-05, global_step: 410, interval_runtime: 38.6466, interval_samples_per_second: 0.207, interval_steps_per_second: 0.259, epoch: 1.0847[0m
[32m[2022-08-31 17:11:39,303] [    INFO][0m - loss: 1.56100721, learning_rate: 2.8333333333333332e-05, global_step: 420, interval_runtime: 3.5352, interval_samples_per_second: 2.263, interval_steps_per_second: 2.829, epoch: 1.1111[0m
[32m[2022-08-31 17:11:42,905] [    INFO][0m - loss: 1.5444581, learning_rate: 2.8293650793650796e-05, global_step: 430, interval_runtime: 3.603, interval_samples_per_second: 2.22, interval_steps_per_second: 2.775, epoch: 1.1376[0m
[32m[2022-08-31 17:11:46,433] [    INFO][0m - loss: 1.47110043, learning_rate: 2.8253968253968253e-05, global_step: 440, interval_runtime: 3.5276, interval_samples_per_second: 2.268, interval_steps_per_second: 2.835, epoch: 1.164[0m
[32m[2022-08-31 17:11:49,986] [    INFO][0m - loss: 1.71892376, learning_rate: 2.8214285714285714e-05, global_step: 450, interval_runtime: 3.5519, interval_samples_per_second: 2.252, interval_steps_per_second: 2.815, epoch: 1.1905[0m
[32m[2022-08-31 17:11:53,619] [    INFO][0m - loss: 1.53718939, learning_rate: 2.8174603174603175e-05, global_step: 460, interval_runtime: 3.6337, interval_samples_per_second: 2.202, interval_steps_per_second: 2.752, epoch: 1.2169[0m
[32m[2022-08-31 17:11:57,216] [    INFO][0m - loss: 1.74545746, learning_rate: 2.8134920634920636e-05, global_step: 470, interval_runtime: 3.5968, interval_samples_per_second: 2.224, interval_steps_per_second: 2.78, epoch: 1.2434[0m
[32m[2022-08-31 17:12:00,900] [    INFO][0m - loss: 1.71188889, learning_rate: 2.8095238095238096e-05, global_step: 480, interval_runtime: 3.6843, interval_samples_per_second: 2.171, interval_steps_per_second: 2.714, epoch: 1.2698[0m
[32m[2022-08-31 17:12:04,573] [    INFO][0m - loss: 1.8654974, learning_rate: 2.8055555555555557e-05, global_step: 490, interval_runtime: 3.6727, interval_samples_per_second: 2.178, interval_steps_per_second: 2.723, epoch: 1.2963[0m
[32m[2022-08-31 17:12:08,424] [    INFO][0m - loss: 1.95640469, learning_rate: 2.8015873015873015e-05, global_step: 500, interval_runtime: 3.8516, interval_samples_per_second: 2.077, interval_steps_per_second: 2.596, epoch: 1.3228[0m
[32m[2022-08-31 17:12:08,425] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 17:12:08,425] [    INFO][0m -   Num examples = 1373[0m
[32m[2022-08-31 17:12:08,425] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 17:12:08,426] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 17:12:08,426] [    INFO][0m -   Total prediction steps = 43[0m
[32m[2022-08-31 17:12:45,757] [    INFO][0m - eval_loss: 2.035566568374634, eval_accuracy: 0.5018208302986161, eval_runtime: 37.3313, eval_samples_per_second: 36.779, eval_steps_per_second: 1.152, epoch: 1.3228[0m
[32m[2022-08-31 17:12:45,758] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-500[0m
[32m[2022-08-31 17:12:45,758] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 17:12:46,802] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-500/tokenizer_config.json[0m
[32m[2022-08-31 17:12:46,802] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-500/special_tokens_map.json[0m
[32m[2022-08-31 17:12:52,392] [    INFO][0m - loss: 2.0145916, learning_rate: 2.797619047619048e-05, global_step: 510, interval_runtime: 43.9681, interval_samples_per_second: 0.182, interval_steps_per_second: 0.227, epoch: 1.3492[0m
[32m[2022-08-31 17:12:56,705] [    INFO][0m - loss: 1.59690247, learning_rate: 2.7936507936507936e-05, global_step: 520, interval_runtime: 3.9736, interval_samples_per_second: 2.013, interval_steps_per_second: 2.517, epoch: 1.3757[0m
[32m[2022-08-31 17:13:00,644] [    INFO][0m - loss: 1.32281561, learning_rate: 2.7896825396825397e-05, global_step: 530, interval_runtime: 4.2775, interval_samples_per_second: 1.87, interval_steps_per_second: 2.338, epoch: 1.4021[0m
[32m[2022-08-31 17:13:04,555] [    INFO][0m - loss: 1.79310131, learning_rate: 2.7857142857142858e-05, global_step: 540, interval_runtime: 3.9118, interval_samples_per_second: 2.045, interval_steps_per_second: 2.556, epoch: 1.4286[0m
[32m[2022-08-31 17:13:08,351] [    INFO][0m - loss: 1.75726681, learning_rate: 2.781746031746032e-05, global_step: 550, interval_runtime: 3.7955, interval_samples_per_second: 2.108, interval_steps_per_second: 2.635, epoch: 1.455[0m
[32m[2022-08-31 17:13:12,335] [    INFO][0m - loss: 1.64904785, learning_rate: 2.777777777777778e-05, global_step: 560, interval_runtime: 3.9841, interval_samples_per_second: 2.008, interval_steps_per_second: 2.51, epoch: 1.4815[0m
[32m[2022-08-31 17:13:16,374] [    INFO][0m - loss: 1.72260246, learning_rate: 2.773809523809524e-05, global_step: 570, interval_runtime: 4.0391, interval_samples_per_second: 1.981, interval_steps_per_second: 2.476, epoch: 1.5079[0m
[32m[2022-08-31 17:13:20,327] [    INFO][0m - loss: 1.67506161, learning_rate: 2.7698412698412697e-05, global_step: 580, interval_runtime: 3.9524, interval_samples_per_second: 2.024, interval_steps_per_second: 2.53, epoch: 1.5344[0m
[32m[2022-08-31 17:13:24,201] [    INFO][0m - loss: 1.74941463, learning_rate: 2.765873015873016e-05, global_step: 590, interval_runtime: 3.8746, interval_samples_per_second: 2.065, interval_steps_per_second: 2.581, epoch: 1.5608[0m
[32m[2022-08-31 17:13:28,196] [    INFO][0m - loss: 1.69477673, learning_rate: 2.761904761904762e-05, global_step: 600, interval_runtime: 3.9951, interval_samples_per_second: 2.002, interval_steps_per_second: 2.503, epoch: 1.5873[0m
[32m[2022-08-31 17:13:28,198] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 17:13:28,198] [    INFO][0m -   Num examples = 1373[0m
[32m[2022-08-31 17:13:28,198] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 17:13:28,198] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 17:13:28,198] [    INFO][0m -   Total prediction steps = 43[0m
[32m[2022-08-31 17:14:10,897] [    INFO][0m - eval_loss: 1.9774668216705322, eval_accuracy: 0.5207574654042243, eval_runtime: 42.6987, eval_samples_per_second: 32.156, eval_steps_per_second: 1.007, epoch: 1.5873[0m
[32m[2022-08-31 17:14:10,898] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-600[0m
[32m[2022-08-31 17:14:10,898] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 17:14:11,954] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-600/tokenizer_config.json[0m
[32m[2022-08-31 17:14:11,955] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-600/special_tokens_map.json[0m
[32m[2022-08-31 17:14:17,832] [    INFO][0m - loss: 1.71247597, learning_rate: 2.757936507936508e-05, global_step: 610, interval_runtime: 49.6355, interval_samples_per_second: 0.161, interval_steps_per_second: 0.201, epoch: 1.6138[0m
[32m[2022-08-31 17:14:22,022] [    INFO][0m - loss: 1.89846878, learning_rate: 2.753968253968254e-05, global_step: 620, interval_runtime: 4.1902, interval_samples_per_second: 1.909, interval_steps_per_second: 2.387, epoch: 1.6402[0m
[32m[2022-08-31 17:14:26,219] [    INFO][0m - loss: 1.55303993, learning_rate: 2.75e-05, global_step: 630, interval_runtime: 4.1974, interval_samples_per_second: 1.906, interval_steps_per_second: 2.382, epoch: 1.6667[0m
[32m[2022-08-31 17:14:30,468] [    INFO][0m - loss: 1.84568996, learning_rate: 2.7460317460317462e-05, global_step: 640, interval_runtime: 4.2485, interval_samples_per_second: 1.883, interval_steps_per_second: 2.354, epoch: 1.6931[0m
[32m[2022-08-31 17:14:38,924] [    INFO][0m - loss: 1.70499649, learning_rate: 2.7420634920634922e-05, global_step: 650, interval_runtime: 4.2956, interval_samples_per_second: 1.862, interval_steps_per_second: 2.328, epoch: 1.7196[0m
[32m[2022-08-31 17:14:43,377] [    INFO][0m - loss: 1.92111454, learning_rate: 2.738095238095238e-05, global_step: 660, interval_runtime: 8.6131, interval_samples_per_second: 0.929, interval_steps_per_second: 1.161, epoch: 1.746[0m
[32m[2022-08-31 17:14:47,702] [    INFO][0m - loss: 1.67467022, learning_rate: 2.7341269841269844e-05, global_step: 670, interval_runtime: 4.3253, interval_samples_per_second: 1.85, interval_steps_per_second: 2.312, epoch: 1.7725[0m
[32m[2022-08-31 17:14:52,032] [    INFO][0m - loss: 1.8762722, learning_rate: 2.73015873015873e-05, global_step: 680, interval_runtime: 4.3295, interval_samples_per_second: 1.848, interval_steps_per_second: 2.31, epoch: 1.7989[0m
[32m[2022-08-31 17:14:56,475] [    INFO][0m - loss: 1.71531124, learning_rate: 2.7261904761904762e-05, global_step: 690, interval_runtime: 4.4434, interval_samples_per_second: 1.8, interval_steps_per_second: 2.251, epoch: 1.8254[0m
[32m[2022-08-31 17:15:00,866] [    INFO][0m - loss: 1.5500762, learning_rate: 2.7222222222222223e-05, global_step: 700, interval_runtime: 4.3906, interval_samples_per_second: 1.822, interval_steps_per_second: 2.278, epoch: 1.8519[0m
[32m[2022-08-31 17:15:00,866] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 17:15:00,867] [    INFO][0m -   Num examples = 1373[0m
[32m[2022-08-31 17:15:00,867] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 17:15:00,867] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 17:15:00,867] [    INFO][0m -   Total prediction steps = 43[0m
[32m[2022-08-31 17:15:47,641] [    INFO][0m - eval_loss: 2.0116100311279297, eval_accuracy: 0.5236707938820102, eval_runtime: 46.7743, eval_samples_per_second: 29.354, eval_steps_per_second: 0.919, epoch: 1.8519[0m
[32m[2022-08-31 17:15:47,642] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-700[0m
[32m[2022-08-31 17:15:47,642] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 17:15:48,713] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-700/tokenizer_config.json[0m
[32m[2022-08-31 17:15:48,713] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-700/special_tokens_map.json[0m
[32m[2022-08-31 17:15:55,018] [    INFO][0m - loss: 2.13487568, learning_rate: 2.7182539682539684e-05, global_step: 710, interval_runtime: 54.152, interval_samples_per_second: 0.148, interval_steps_per_second: 0.185, epoch: 1.8783[0m
[32m[2022-08-31 17:15:59,606] [    INFO][0m - loss: 1.4657691, learning_rate: 2.7142857142857144e-05, global_step: 720, interval_runtime: 4.5881, interval_samples_per_second: 1.744, interval_steps_per_second: 2.18, epoch: 1.9048[0m
[32m[2022-08-31 17:16:04,212] [    INFO][0m - loss: 2.11249256, learning_rate: 2.7103174603174605e-05, global_step: 730, interval_runtime: 4.6064, interval_samples_per_second: 1.737, interval_steps_per_second: 2.171, epoch: 1.9312[0m
[32m[2022-08-31 17:16:08,846] [    INFO][0m - loss: 1.58489656, learning_rate: 2.7063492063492062e-05, global_step: 740, interval_runtime: 4.6338, interval_samples_per_second: 1.726, interval_steps_per_second: 2.158, epoch: 1.9577[0m
[32m[2022-08-31 17:16:13,536] [    INFO][0m - loss: 1.78765965, learning_rate: 2.7023809523809527e-05, global_step: 750, interval_runtime: 4.6897, interval_samples_per_second: 1.706, interval_steps_per_second: 2.132, epoch: 1.9841[0m
[32m[2022-08-31 17:16:18,353] [    INFO][0m - loss: 1.18521757, learning_rate: 2.6984126984126984e-05, global_step: 760, interval_runtime: 4.817, interval_samples_per_second: 1.661, interval_steps_per_second: 2.076, epoch: 2.0106[0m
[32m[2022-08-31 17:16:23,195] [    INFO][0m - loss: 0.84699068, learning_rate: 2.6944444444444445e-05, global_step: 770, interval_runtime: 4.8421, interval_samples_per_second: 1.652, interval_steps_per_second: 2.065, epoch: 2.037[0m
[32m[2022-08-31 17:16:27,984] [    INFO][0m - loss: 1.07917891, learning_rate: 2.6904761904761905e-05, global_step: 780, interval_runtime: 4.7896, interval_samples_per_second: 1.67, interval_steps_per_second: 2.088, epoch: 2.0635[0m
[32m[2022-08-31 17:16:32,837] [    INFO][0m - loss: 1.14326611, learning_rate: 2.6865079365079366e-05, global_step: 790, interval_runtime: 4.8523, interval_samples_per_second: 1.649, interval_steps_per_second: 2.061, epoch: 2.0899[0m
[32m[2022-08-31 17:16:37,659] [    INFO][0m - loss: 0.92890072, learning_rate: 2.6825396825396827e-05, global_step: 800, interval_runtime: 4.822, interval_samples_per_second: 1.659, interval_steps_per_second: 2.074, epoch: 2.1164[0m
[32m[2022-08-31 17:16:37,659] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 17:16:37,660] [    INFO][0m -   Num examples = 1373[0m
[32m[2022-08-31 17:16:37,660] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 17:16:37,660] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 17:16:37,660] [    INFO][0m -   Total prediction steps = 43[0m
[32m[2022-08-31 17:17:30,208] [    INFO][0m - eval_loss: 2.030925989151001, eval_accuracy: 0.5651857246904588, eval_runtime: 52.5472, eval_samples_per_second: 26.129, eval_steps_per_second: 0.818, epoch: 2.1164[0m
[32m[2022-08-31 17:17:30,208] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-800[0m
[32m[2022-08-31 17:17:30,208] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 17:17:31,364] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-800/tokenizer_config.json[0m
[32m[2022-08-31 17:17:31,365] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-800/special_tokens_map.json[0m
[32m[2022-08-31 17:17:40,685] [    INFO][0m - loss: 1.114645, learning_rate: 2.6785714285714288e-05, global_step: 810, interval_runtime: 60.4863, interval_samples_per_second: 0.132, interval_steps_per_second: 0.165, epoch: 2.1429[0m
[32m[2022-08-31 17:17:45,564] [    INFO][0m - loss: 1.15444307, learning_rate: 2.6746031746031745e-05, global_step: 820, interval_runtime: 7.4189, interval_samples_per_second: 1.078, interval_steps_per_second: 1.348, epoch: 2.1693[0m
[32m[2022-08-31 17:17:50,559] [    INFO][0m - loss: 1.13179483, learning_rate: 2.670634920634921e-05, global_step: 830, interval_runtime: 4.9952, interval_samples_per_second: 1.602, interval_steps_per_second: 2.002, epoch: 2.1958[0m
[32m[2022-08-31 17:17:55,538] [    INFO][0m - loss: 1.03354139, learning_rate: 2.6666666666666667e-05, global_step: 840, interval_runtime: 4.9785, interval_samples_per_second: 1.607, interval_steps_per_second: 2.009, epoch: 2.2222[0m
[32m[2022-08-31 17:18:00,639] [    INFO][0m - loss: 1.20296812, learning_rate: 2.6626984126984127e-05, global_step: 850, interval_runtime: 5.1012, interval_samples_per_second: 1.568, interval_steps_per_second: 1.96, epoch: 2.2487[0m
[32m[2022-08-31 17:18:05,752] [    INFO][0m - loss: 1.0923254, learning_rate: 2.6587301587301588e-05, global_step: 860, interval_runtime: 5.1128, interval_samples_per_second: 1.565, interval_steps_per_second: 1.956, epoch: 2.2751[0m
[32m[2022-08-31 17:18:10,928] [    INFO][0m - loss: 1.16058311, learning_rate: 2.654761904761905e-05, global_step: 870, interval_runtime: 5.1756, interval_samples_per_second: 1.546, interval_steps_per_second: 1.932, epoch: 2.3016[0m
[32m[2022-08-31 17:18:16,041] [    INFO][0m - loss: 1.02819462, learning_rate: 2.650793650793651e-05, global_step: 880, interval_runtime: 5.1139, interval_samples_per_second: 1.564, interval_steps_per_second: 1.955, epoch: 2.328[0m
[32m[2022-08-31 17:18:21,237] [    INFO][0m - loss: 0.92013569, learning_rate: 2.646825396825397e-05, global_step: 890, interval_runtime: 5.1959, interval_samples_per_second: 1.54, interval_steps_per_second: 1.925, epoch: 2.3545[0m
[32m[2022-08-31 17:18:26,394] [    INFO][0m - loss: 1.04477968, learning_rate: 2.6428571428571428e-05, global_step: 900, interval_runtime: 5.1563, interval_samples_per_second: 1.552, interval_steps_per_second: 1.939, epoch: 2.381[0m
[32m[2022-08-31 17:18:26,395] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 17:18:26,395] [    INFO][0m -   Num examples = 1373[0m
[32m[2022-08-31 17:18:26,395] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 17:18:26,395] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 17:18:26,396] [    INFO][0m -   Total prediction steps = 43[0m
[32m[2022-08-31 17:19:23,962] [    INFO][0m - eval_loss: 2.0735769271850586, eval_accuracy: 0.5375091041514931, eval_runtime: 57.5649, eval_samples_per_second: 23.851, eval_steps_per_second: 0.747, epoch: 2.381[0m
[32m[2022-08-31 17:19:23,963] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-900[0m
[32m[2022-08-31 17:19:23,963] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 17:19:24,973] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-900/tokenizer_config.json[0m
[32m[2022-08-31 17:19:24,973] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-900/special_tokens_map.json[0m
[32m[2022-08-31 17:19:32,031] [    INFO][0m - loss: 1.01064396, learning_rate: 2.6388888888888892e-05, global_step: 910, interval_runtime: 65.6376, interval_samples_per_second: 0.122, interval_steps_per_second: 0.152, epoch: 2.4074[0m
[32m[2022-08-31 17:19:37,480] [    INFO][0m - loss: 1.07724619, learning_rate: 2.634920634920635e-05, global_step: 920, interval_runtime: 5.449, interval_samples_per_second: 1.468, interval_steps_per_second: 1.835, epoch: 2.4339[0m
[32m[2022-08-31 17:19:43,015] [    INFO][0m - loss: 1.22042999, learning_rate: 2.630952380952381e-05, global_step: 930, interval_runtime: 5.5347, interval_samples_per_second: 1.445, interval_steps_per_second: 1.807, epoch: 2.4603[0m
[32m[2022-08-31 17:19:48,612] [    INFO][0m - loss: 1.00473099, learning_rate: 2.626984126984127e-05, global_step: 940, interval_runtime: 5.5967, interval_samples_per_second: 1.429, interval_steps_per_second: 1.787, epoch: 2.4868[0m
[32m[2022-08-31 17:19:54,014] [    INFO][0m - loss: 1.38511801, learning_rate: 2.623015873015873e-05, global_step: 950, interval_runtime: 5.4023, interval_samples_per_second: 1.481, interval_steps_per_second: 1.851, epoch: 2.5132[0m
[32m[2022-08-31 17:19:59,455] [    INFO][0m - loss: 1.14462852, learning_rate: 2.6190476190476192e-05, global_step: 960, interval_runtime: 5.4414, interval_samples_per_second: 1.47, interval_steps_per_second: 1.838, epoch: 2.5397[0m
[32m[2022-08-31 17:20:04,867] [    INFO][0m - loss: 1.04213495, learning_rate: 2.6150793650793653e-05, global_step: 970, interval_runtime: 5.4116, interval_samples_per_second: 1.478, interval_steps_per_second: 1.848, epoch: 2.5661[0m
[32m[2022-08-31 17:20:10,325] [    INFO][0m - loss: 1.00367298, learning_rate: 2.611111111111111e-05, global_step: 980, interval_runtime: 5.4578, interval_samples_per_second: 1.466, interval_steps_per_second: 1.832, epoch: 2.5926[0m
[32m[2022-08-31 17:20:15,828] [    INFO][0m - loss: 1.12793455, learning_rate: 2.607142857142857e-05, global_step: 990, interval_runtime: 5.503, interval_samples_per_second: 1.454, interval_steps_per_second: 1.817, epoch: 2.619[0m
[32m[2022-08-31 17:20:21,399] [    INFO][0m - loss: 1.18662815, learning_rate: 2.6031746031746032e-05, global_step: 1000, interval_runtime: 5.5718, interval_samples_per_second: 1.436, interval_steps_per_second: 1.795, epoch: 2.6455[0m
[32m[2022-08-31 17:20:21,400] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 17:20:21,400] [    INFO][0m -   Num examples = 1373[0m
[32m[2022-08-31 17:20:21,400] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 17:20:21,400] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 17:20:21,400] [    INFO][0m -   Total prediction steps = 43[0m
[32m[2022-08-31 17:21:24,253] [    INFO][0m - eval_loss: 2.0519609451293945, eval_accuracy: 0.528040786598689, eval_runtime: 62.8519, eval_samples_per_second: 21.845, eval_steps_per_second: 0.684, epoch: 2.6455[0m
[32m[2022-08-31 17:21:24,253] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-1000[0m
[32m[2022-08-31 17:21:24,254] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 17:21:25,376] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-1000/tokenizer_config.json[0m
[32m[2022-08-31 17:21:25,377] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-1000/special_tokens_map.json[0m
[32m[2022-08-31 17:21:32,693] [    INFO][0m - loss: 1.21521931, learning_rate: 2.5992063492063493e-05, global_step: 1010, interval_runtime: 71.293, interval_samples_per_second: 0.112, interval_steps_per_second: 0.14, epoch: 2.672[0m
[32m[2022-08-31 17:21:38,391] [    INFO][0m - loss: 1.22747707, learning_rate: 2.5952380952380953e-05, global_step: 1020, interval_runtime: 5.6984, interval_samples_per_second: 1.404, interval_steps_per_second: 1.755, epoch: 2.6984[0m
[32m[2022-08-31 17:21:44,117] [    INFO][0m - loss: 1.22034903, learning_rate: 2.591269841269841e-05, global_step: 1030, interval_runtime: 5.7262, interval_samples_per_second: 1.397, interval_steps_per_second: 1.746, epoch: 2.7249[0m
[32m[2022-08-31 17:21:50,147] [    INFO][0m - loss: 1.27183685, learning_rate: 2.5873015873015875e-05, global_step: 1040, interval_runtime: 5.7602, interval_samples_per_second: 1.389, interval_steps_per_second: 1.736, epoch: 2.7513[0m
[32m[2022-08-31 17:21:56,047] [    INFO][0m - loss: 1.29419355, learning_rate: 2.5833333333333336e-05, global_step: 1050, interval_runtime: 6.17, interval_samples_per_second: 1.297, interval_steps_per_second: 1.621, epoch: 2.7778[0m
[32m[2022-08-31 17:22:01,957] [    INFO][0m - loss: 1.25833235, learning_rate: 2.5793650793650793e-05, global_step: 1060, interval_runtime: 5.9096, interval_samples_per_second: 1.354, interval_steps_per_second: 1.692, epoch: 2.8042[0m
[32m[2022-08-31 17:22:07,771] [    INFO][0m - loss: 1.2027154, learning_rate: 2.5753968253968254e-05, global_step: 1070, interval_runtime: 5.814, interval_samples_per_second: 1.376, interval_steps_per_second: 1.72, epoch: 2.8307[0m
[32m[2022-08-31 17:22:13,693] [    INFO][0m - loss: 1.13933611, learning_rate: 2.5714285714285714e-05, global_step: 1080, interval_runtime: 5.9216, interval_samples_per_second: 1.351, interval_steps_per_second: 1.689, epoch: 2.8571[0m
[32m[2022-08-31 17:22:19,647] [    INFO][0m - loss: 1.01160898, learning_rate: 2.5674603174603175e-05, global_step: 1090, interval_runtime: 5.9544, interval_samples_per_second: 1.344, interval_steps_per_second: 1.679, epoch: 2.8836[0m
[32m[2022-08-31 17:22:25,729] [    INFO][0m - loss: 1.17013063, learning_rate: 2.5634920634920636e-05, global_step: 1100, interval_runtime: 6.0819, interval_samples_per_second: 1.315, interval_steps_per_second: 1.644, epoch: 2.9101[0m
[32m[2022-08-31 17:22:25,730] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 17:22:25,730] [    INFO][0m -   Num examples = 1373[0m
[32m[2022-08-31 17:22:25,731] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 17:22:25,731] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 17:22:25,731] [    INFO][0m -   Total prediction steps = 43[0m
[32m[2022-08-31 17:23:32,236] [    INFO][0m - eval_loss: 1.9929590225219727, eval_accuracy: 0.5447924253459577, eval_runtime: 66.5049, eval_samples_per_second: 20.645, eval_steps_per_second: 0.647, epoch: 2.9101[0m
[32m[2022-08-31 17:23:32,237] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-1100[0m
[32m[2022-08-31 17:23:32,237] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 17:23:33,232] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-1100/tokenizer_config.json[0m
[32m[2022-08-31 17:23:33,232] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-1100/special_tokens_map.json[0m
[32m[2022-08-31 17:23:41,086] [    INFO][0m - loss: 1.04020567, learning_rate: 2.5595238095238093e-05, global_step: 1110, interval_runtime: 75.3569, interval_samples_per_second: 0.106, interval_steps_per_second: 0.133, epoch: 2.9365[0m
[32m[2022-08-31 17:23:47,225] [    INFO][0m - loss: 0.96024818, learning_rate: 2.5555555555555557e-05, global_step: 1120, interval_runtime: 6.1392, interval_samples_per_second: 1.303, interval_steps_per_second: 1.629, epoch: 2.963[0m
[32m[2022-08-31 17:23:52,990] [    INFO][0m - loss: 1.13494959, learning_rate: 2.5515873015873018e-05, global_step: 1130, interval_runtime: 5.7645, interval_samples_per_second: 1.388, interval_steps_per_second: 1.735, epoch: 2.9894[0m
[32m[2022-08-31 17:23:59,722] [    INFO][0m - loss: 1.01026516, learning_rate: 2.5476190476190476e-05, global_step: 1140, interval_runtime: 6.7321, interval_samples_per_second: 1.188, interval_steps_per_second: 1.485, epoch: 3.0159[0m
[32m[2022-08-31 17:24:05,958] [    INFO][0m - loss: 0.6140758, learning_rate: 2.5436507936507936e-05, global_step: 1150, interval_runtime: 6.2357, interval_samples_per_second: 1.283, interval_steps_per_second: 1.604, epoch: 3.0423[0m
[32m[2022-08-31 17:24:12,103] [    INFO][0m - loss: 0.70122128, learning_rate: 2.5396825396825397e-05, global_step: 1160, interval_runtime: 6.1456, interval_samples_per_second: 1.302, interval_steps_per_second: 1.627, epoch: 3.0688[0m
[32m[2022-08-31 17:24:18,284] [    INFO][0m - loss: 0.84951792, learning_rate: 2.5357142857142858e-05, global_step: 1170, interval_runtime: 6.1805, interval_samples_per_second: 1.294, interval_steps_per_second: 1.618, epoch: 3.0952[0m
[32m[2022-08-31 17:24:24,585] [    INFO][0m - loss: 0.75621433, learning_rate: 2.531746031746032e-05, global_step: 1180, interval_runtime: 6.3009, interval_samples_per_second: 1.27, interval_steps_per_second: 1.587, epoch: 3.1217[0m
[32m[2022-08-31 17:24:30,871] [    INFO][0m - loss: 0.70605769, learning_rate: 2.5277777777777776e-05, global_step: 1190, interval_runtime: 6.2862, interval_samples_per_second: 1.273, interval_steps_per_second: 1.591, epoch: 3.1481[0m
[32m[2022-08-31 17:24:37,162] [    INFO][0m - loss: 0.65423908, learning_rate: 2.523809523809524e-05, global_step: 1200, interval_runtime: 6.2909, interval_samples_per_second: 1.272, interval_steps_per_second: 1.59, epoch: 3.1746[0m
[32m[2022-08-31 17:24:37,163] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 17:24:37,163] [    INFO][0m -   Num examples = 1373[0m
[32m[2022-08-31 17:24:37,163] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 17:24:37,163] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 17:24:37,163] [    INFO][0m -   Total prediction steps = 43[0m
[32m[2022-08-31 17:25:50,156] [    INFO][0m - eval_loss: 2.154998540878296, eval_accuracy: 0.5302257829570284, eval_runtime: 72.9922, eval_samples_per_second: 18.81, eval_steps_per_second: 0.589, epoch: 3.1746[0m
[32m[2022-08-31 17:25:50,156] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-1200[0m
[32m[2022-08-31 17:25:50,157] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 17:25:51,178] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-1200/tokenizer_config.json[0m
[32m[2022-08-31 17:25:51,178] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-1200/special_tokens_map.json[0m
[32m[2022-08-31 17:25:53,102] [    INFO][0m - 
Training completed. 
[0m
[32m[2022-08-31 17:25:53,102] [    INFO][0m - Loading best model from ./checkpoints/checkpoint-800 (score: 0.5651857246904588).[0m
[32m[2022-08-31 17:25:53,901] [    INFO][0m - train_runtime: 1081.2467, train_samples_per_second: 55.935, train_steps_per_second: 6.992, train_loss: 1.9880492639541627, epoch: 3.1746[0m
[32m[2022-08-31 17:25:53,902] [    INFO][0m - Saving model checkpoint to ./checkpoints/[0m
[32m[2022-08-31 17:25:53,902] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 17:25:54,986] [    INFO][0m - tokenizer config file saved in ./checkpoints/tokenizer_config.json[0m
[32m[2022-08-31 17:25:54,987] [    INFO][0m - Special tokens file saved in ./checkpoints/special_tokens_map.json[0m
[32m[2022-08-31 17:25:54,988] [    INFO][0m - ***** train metrics *****[0m
[32m[2022-08-31 17:25:54,988] [    INFO][0m -   epoch                    =     3.1746[0m
[32m[2022-08-31 17:25:54,988] [    INFO][0m -   train_loss               =      1.988[0m
[32m[2022-08-31 17:25:54,988] [    INFO][0m -   train_runtime            = 0:18:01.24[0m
[32m[2022-08-31 17:25:54,988] [    INFO][0m -   train_samples_per_second =     55.935[0m
[32m[2022-08-31 17:25:54,988] [    INFO][0m -   train_steps_per_second   =      6.992[0m
[32m[2022-08-31 17:25:54,993] [    INFO][0m - ***** Running Prediction *****[0m
[32m[2022-08-31 17:25:54,994] [    INFO][0m -   Num examples = 1749[0m
[32m[2022-08-31 17:25:54,994] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 17:25:54,994] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 17:25:54,994] [    INFO][0m -   Total prediction steps = 55[0m
[32m[2022-08-31 17:27:30,366] [    INFO][0m - ***** test metrics *****[0m
[32m[2022-08-31 17:27:30,367] [    INFO][0m -   test_accuracy           =     0.5603[0m
[32m[2022-08-31 17:27:30,367] [    INFO][0m -   test_loss               =     1.9711[0m
[32m[2022-08-31 17:27:30,367] [    INFO][0m -   test_runtime            = 0:01:35.37[0m
[32m[2022-08-31 17:27:30,367] [    INFO][0m -   test_samples_per_second =     18.339[0m
[32m[2022-08-31 17:27:30,367] [    INFO][0m -   test_steps_per_second   =      0.577[0m
[32m[2022-08-31 17:27:30,367] [    INFO][0m - ***** Running Prediction *****[0m
[32m[2022-08-31 17:27:30,367] [    INFO][0m -   Num examples = 2600[0m
[32m[2022-08-31 17:27:30,367] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 17:27:30,368] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 17:27:30,368] [    INFO][0m -   Total prediction steps = 82[0m
[32m[2022-08-31 17:30:03,327] [    INFO][0m - Predictions for iflytekf saved to ./fewclue_submit_examples.[0m
run.sh: line 64: --model_name_or_path: command not found
