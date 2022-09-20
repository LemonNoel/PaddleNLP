[33m[2022-08-31 15:21:35,058] [ WARNING][0m - evaluation_strategy reset to IntervalStrategy.STEPS for do_eval is True. you can also set evaluation_strategy='epoch'.[0m
[32m[2022-08-31 15:21:35,059] [    INFO][0m - The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).[0m
[32m[2022-08-31 15:21:35,059] [    INFO][0m - ============================================================[0m
[32m[2022-08-31 15:21:35,059] [    INFO][0m -      Model Configuration Arguments      [0m
[32m[2022-08-31 15:21:35,059] [    INFO][0m - paddle commit id              :65f388690048d0965ec7d3b43fb6ee9d8c6dee7c[0m
[32m[2022-08-31 15:21:35,059] [    INFO][0m - do_save                       :True[0m
[32m[2022-08-31 15:21:35,059] [    INFO][0m - do_test                       :True[0m
[32m[2022-08-31 15:21:35,059] [    INFO][0m - early_stop_patience           :4[0m
[32m[2022-08-31 15:21:35,059] [    INFO][0m - export_type                   :paddle[0m
[32m[2022-08-31 15:21:35,060] [    INFO][0m - model_name_or_path            :ernie-3.0-base-zh[0m
[32m[2022-08-31 15:21:35,060] [    INFO][0m - [0m
[32m[2022-08-31 15:21:35,060] [    INFO][0m - ============================================================[0m
[32m[2022-08-31 15:21:35,060] [    INFO][0m -       Data Configuration Arguments      [0m
[32m[2022-08-31 15:21:35,060] [    INFO][0m - paddle commit id              :65f388690048d0965ec7d3b43fb6ee9d8c6dee7c[0m
[32m[2022-08-31 15:21:35,060] [    INFO][0m - encoder_hidden_size           :200[0m
[32m[2022-08-31 15:21:35,060] [    INFO][0m - prompt                        :“{'text':'text_a'}”和“{'text':'text_b'}”之间的逻辑关系是{'mask'}{'mask'}。[0m
[32m[2022-08-31 15:21:35,060] [    INFO][0m - soft_encoder                  :lstm[0m
[32m[2022-08-31 15:21:35,060] [    INFO][0m - split_id                      :few_all[0m
[32m[2022-08-31 15:21:35,060] [    INFO][0m - task_name                     :ocnli[0m
[32m[2022-08-31 15:21:35,060] [    INFO][0m - [0m
[32m[2022-08-31 15:21:35,061] [    INFO][0m - Already cached /ssd2/wanghuijuan03/.paddlenlp/models/ernie-3.0-base-zh/ernie_3.0_base_zh.pdparams[0m
W0831 15:21:35.062707 58725 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 11.2
W0831 15:21:35.067006 58725 gpu_resources.cc:91] device: 0, cuDNN Version: 8.1.
[32m[2022-08-31 15:21:38,126] [    INFO][0m - Already cached /ssd2/wanghuijuan03/.paddlenlp/models/ernie-3.0-base-zh/ernie_3.0_base_zh_vocab.txt[0m
[32m[2022-08-31 15:21:38,151] [    INFO][0m - tokenizer config file saved in /ssd2/wanghuijuan03/.paddlenlp/models/ernie-3.0-base-zh/tokenizer_config.json[0m
[32m[2022-08-31 15:21:38,152] [    INFO][0m - Special tokens file saved in /ssd2/wanghuijuan03/.paddlenlp/models/ernie-3.0-base-zh/special_tokens_map.json[0m
[32m[2022-08-31 15:21:38,153] [    INFO][0m - Using template: [{'add_prefix_space': '', 'hard': '“'}, {'add_prefix_space': '', 'text': 'text_a'}, {'add_prefix_space': '', 'hard': '”和“'}, {'add_prefix_space': '', 'text': 'text_b'}, {'add_prefix_space': '', 'hard': '”之间的逻辑关系是'}, {'add_prefix_space': '', 'mask': None}, {'add_prefix_space': '', 'mask': None}, {'add_prefix_space': '', 'hard': '。'}][0m
[32m[2022-08-31 15:21:38,155] [    INFO][0m - {'contradiction': 0, 'entailment': 1, 'neutral': 2}[0m
2022-08-31 15:21:38,157 INFO [download.py:119] unique_endpoints {''}
[32m[2022-08-31 15:21:39,409] [    INFO][0m - ============================================================[0m
[32m[2022-08-31 15:21:39,410] [    INFO][0m -     Training Configuration Arguments    [0m
[32m[2022-08-31 15:21:39,410] [    INFO][0m - paddle commit id              :65f388690048d0965ec7d3b43fb6ee9d8c6dee7c[0m
[32m[2022-08-31 15:21:39,410] [    INFO][0m - _no_sync_in_gradient_accumulation:True[0m
[32m[2022-08-31 15:21:39,411] [    INFO][0m - adam_beta1                    :0.9[0m
[32m[2022-08-31 15:21:39,411] [    INFO][0m - adam_beta2                    :0.999[0m
[32m[2022-08-31 15:21:39,411] [    INFO][0m - adam_epsilon                  :1e-08[0m
[32m[2022-08-31 15:21:39,411] [    INFO][0m - alpha_rdrop                   :5.0[0m
[32m[2022-08-31 15:21:39,411] [    INFO][0m - alpha_rgl                     :0.5[0m
[32m[2022-08-31 15:21:39,411] [    INFO][0m - current_device                :gpu:0[0m
[32m[2022-08-31 15:21:39,411] [    INFO][0m - dataloader_drop_last          :False[0m
[32m[2022-08-31 15:21:39,411] [    INFO][0m - dataloader_num_workers        :0[0m
[32m[2022-08-31 15:21:39,411] [    INFO][0m - device                        :gpu[0m
[32m[2022-08-31 15:21:39,411] [    INFO][0m - disable_tqdm                  :True[0m
[32m[2022-08-31 15:21:39,411] [    INFO][0m - do_eval                       :True[0m
[32m[2022-08-31 15:21:39,411] [    INFO][0m - do_export                     :False[0m
[32m[2022-08-31 15:21:39,412] [    INFO][0m - do_predict                    :True[0m
[32m[2022-08-31 15:21:39,412] [    INFO][0m - do_train                      :True[0m
[32m[2022-08-31 15:21:39,412] [    INFO][0m - eval_batch_size               :64[0m
[32m[2022-08-31 15:21:39,412] [    INFO][0m - eval_steps                    :100[0m
[32m[2022-08-31 15:21:39,412] [    INFO][0m - evaluation_strategy           :IntervalStrategy.STEPS[0m
[32m[2022-08-31 15:21:39,412] [    INFO][0m - first_max_length              :None[0m
[32m[2022-08-31 15:21:39,412] [    INFO][0m - fp16                          :False[0m
[32m[2022-08-31 15:21:39,412] [    INFO][0m - fp16_opt_level                :O1[0m
[32m[2022-08-31 15:21:39,412] [    INFO][0m - freeze_dropout                :False[0m
[32m[2022-08-31 15:21:39,412] [    INFO][0m - freeze_plm                    :False[0m
[32m[2022-08-31 15:21:39,412] [    INFO][0m - gradient_accumulation_steps   :1[0m
[32m[2022-08-31 15:21:39,412] [    INFO][0m - greater_is_better             :True[0m
[32m[2022-08-31 15:21:39,412] [    INFO][0m - ignore_data_skip              :False[0m
[32m[2022-08-31 15:21:39,412] [    INFO][0m - label_names                   :None[0m
[32m[2022-08-31 15:21:39,413] [    INFO][0m - learning_rate                 :3e-06[0m
[32m[2022-08-31 15:21:39,413] [    INFO][0m - load_best_model_at_end        :True[0m
[32m[2022-08-31 15:21:39,413] [    INFO][0m - local_process_index           :0[0m
[32m[2022-08-31 15:21:39,413] [    INFO][0m - local_rank                    :-1[0m
[32m[2022-08-31 15:21:39,413] [    INFO][0m - log_level                     :-1[0m
[32m[2022-08-31 15:21:39,413] [    INFO][0m - log_level_replica             :-1[0m
[32m[2022-08-31 15:21:39,413] [    INFO][0m - log_on_each_node              :True[0m
[32m[2022-08-31 15:21:39,413] [    INFO][0m - logging_dir                   :./checkpoints/runs/Aug31_15-21-35_instance-3bwob41y-01[0m
[32m[2022-08-31 15:21:39,413] [    INFO][0m - logging_first_step            :False[0m
[32m[2022-08-31 15:21:39,413] [    INFO][0m - logging_steps                 :10[0m
[32m[2022-08-31 15:21:39,413] [    INFO][0m - logging_strategy              :IntervalStrategy.STEPS[0m
[32m[2022-08-31 15:21:39,413] [    INFO][0m - lr_scheduler_type             :SchedulerType.LINEAR[0m
[32m[2022-08-31 15:21:39,413] [    INFO][0m - max_grad_norm                 :1.0[0m
[32m[2022-08-31 15:21:39,414] [    INFO][0m - max_seq_length                :128[0m
[32m[2022-08-31 15:21:39,414] [    INFO][0m - max_steps                     :-1[0m
[32m[2022-08-31 15:21:39,414] [    INFO][0m - metric_for_best_model         :accuracy[0m
[32m[2022-08-31 15:21:39,414] [    INFO][0m - minimum_eval_times            :None[0m
[32m[2022-08-31 15:21:39,414] [    INFO][0m - no_cuda                       :False[0m
[32m[2022-08-31 15:21:39,414] [    INFO][0m - num_train_epochs              :50.0[0m
[32m[2022-08-31 15:21:39,414] [    INFO][0m - optim                         :OptimizerNames.ADAMW[0m
[32m[2022-08-31 15:21:39,414] [    INFO][0m - other_max_length              :None[0m
[32m[2022-08-31 15:21:39,414] [    INFO][0m - output_dir                    :./checkpoints/[0m
[32m[2022-08-31 15:21:39,414] [    INFO][0m - overwrite_output_dir          :False[0m
[32m[2022-08-31 15:21:39,414] [    INFO][0m - past_index                    :-1[0m
[32m[2022-08-31 15:21:39,414] [    INFO][0m - per_device_eval_batch_size    :64[0m
[32m[2022-08-31 15:21:39,414] [    INFO][0m - per_device_train_batch_size   :8[0m
[32m[2022-08-31 15:21:39,415] [    INFO][0m - ppt_adam_beta1                :0.9[0m
[32m[2022-08-31 15:21:39,415] [    INFO][0m - ppt_adam_beta2                :0.999[0m
[32m[2022-08-31 15:21:39,415] [    INFO][0m - ppt_adam_epsilon              :1e-08[0m
[32m[2022-08-31 15:21:39,415] [    INFO][0m - ppt_learning_rate             :3e-05[0m
[32m[2022-08-31 15:21:39,415] [    INFO][0m - ppt_weight_decay              :0.0[0m
[32m[2022-08-31 15:21:39,415] [    INFO][0m - prediction_loss_only          :False[0m
[32m[2022-08-31 15:21:39,415] [    INFO][0m - process_index                 :0[0m
[32m[2022-08-31 15:21:39,415] [    INFO][0m - remove_unused_columns         :True[0m
[32m[2022-08-31 15:21:39,415] [    INFO][0m - report_to                     :['visualdl'][0m
[32m[2022-08-31 15:21:39,415] [    INFO][0m - resume_from_checkpoint        :None[0m
[32m[2022-08-31 15:21:39,415] [    INFO][0m - run_name                      :./checkpoints/[0m
[32m[2022-08-31 15:21:39,415] [    INFO][0m - save_on_each_node             :False[0m
[32m[2022-08-31 15:21:39,415] [    INFO][0m - save_steps                    :100[0m
[32m[2022-08-31 15:21:39,416] [    INFO][0m - save_strategy                 :IntervalStrategy.STEPS[0m
[32m[2022-08-31 15:21:39,416] [    INFO][0m - save_total_limit              :None[0m
[32m[2022-08-31 15:21:39,416] [    INFO][0m - scale_loss                    :32768[0m
[32m[2022-08-31 15:21:39,416] [    INFO][0m - seed                          :42[0m
[32m[2022-08-31 15:21:39,416] [    INFO][0m - should_log                    :True[0m
[32m[2022-08-31 15:21:39,416] [    INFO][0m - should_save                   :True[0m
[32m[2022-08-31 15:21:39,416] [    INFO][0m - task_type                     :multi-class[0m
[32m[2022-08-31 15:21:39,416] [    INFO][0m - train_batch_size              :8[0m
[32m[2022-08-31 15:21:39,416] [    INFO][0m - truncate_mode                 :tail[0m
[32m[2022-08-31 15:21:39,416] [    INFO][0m - use_rdrop                     :False[0m
[32m[2022-08-31 15:21:39,416] [    INFO][0m - use_rgl                       :False[0m
[32m[2022-08-31 15:21:39,417] [    INFO][0m - warmup_ratio                  :0.0[0m
[32m[2022-08-31 15:21:39,417] [    INFO][0m - warmup_steps                  :0[0m
[32m[2022-08-31 15:21:39,417] [    INFO][0m - weight_decay                  :0.0[0m
[32m[2022-08-31 15:21:39,417] [    INFO][0m - world_size                    :1[0m
[32m[2022-08-31 15:21:39,417] [    INFO][0m - [0m
[32m[2022-08-31 15:21:39,419] [    INFO][0m - ***** Running training *****[0m
[32m[2022-08-31 15:21:39,419] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 15:21:39,419] [    INFO][0m -   Num Epochs = 50[0m
[32m[2022-08-31 15:21:39,420] [    INFO][0m -   Instantaneous batch size per device = 8[0m
[32m[2022-08-31 15:21:39,420] [    INFO][0m -   Total train batch size (w. parallel, distributed & accumulation) = 8[0m
[32m[2022-08-31 15:21:39,420] [    INFO][0m -   Gradient Accumulation steps = 1[0m
[32m[2022-08-31 15:21:39,420] [    INFO][0m -   Total optimization steps = 1000.0[0m
[32m[2022-08-31 15:21:39,420] [    INFO][0m -   Total num train samples = 8000[0m
[32m[2022-08-31 15:21:41,543] [    INFO][0m - loss: 0.8502594, learning_rate: 2.97e-06, global_step: 10, interval_runtime: 2.1221, interval_samples_per_second: 3.77, interval_steps_per_second: 4.712, epoch: 0.5[0m
[32m[2022-08-31 15:21:42,426] [    INFO][0m - loss: 0.67025094, learning_rate: 2.9400000000000002e-06, global_step: 20, interval_runtime: 0.8829, interval_samples_per_second: 9.061, interval_steps_per_second: 11.327, epoch: 1.0[0m
[32m[2022-08-31 15:21:43,376] [    INFO][0m - loss: 0.63618245, learning_rate: 2.91e-06, global_step: 30, interval_runtime: 0.9501, interval_samples_per_second: 8.42, interval_steps_per_second: 10.525, epoch: 1.5[0m
[32m[2022-08-31 15:21:44,256] [    INFO][0m - loss: 0.72451291, learning_rate: 2.88e-06, global_step: 40, interval_runtime: 0.8802, interval_samples_per_second: 9.089, interval_steps_per_second: 11.361, epoch: 2.0[0m
[32m[2022-08-31 15:21:45,214] [    INFO][0m - loss: 0.68271666, learning_rate: 2.85e-06, global_step: 50, interval_runtime: 0.9579, interval_samples_per_second: 8.352, interval_steps_per_second: 10.439, epoch: 2.5[0m
[32m[2022-08-31 15:21:46,094] [    INFO][0m - loss: 0.56053514, learning_rate: 2.82e-06, global_step: 60, interval_runtime: 0.8807, interval_samples_per_second: 9.084, interval_steps_per_second: 11.355, epoch: 3.0[0m
[32m[2022-08-31 15:21:47,043] [    INFO][0m - loss: 0.48508258, learning_rate: 2.7900000000000004e-06, global_step: 70, interval_runtime: 0.9485, interval_samples_per_second: 8.434, interval_steps_per_second: 10.543, epoch: 3.5[0m
[32m[2022-08-31 15:21:47,941] [    INFO][0m - loss: 0.70130019, learning_rate: 2.7600000000000003e-06, global_step: 80, interval_runtime: 0.8982, interval_samples_per_second: 8.907, interval_steps_per_second: 11.133, epoch: 4.0[0m
[32m[2022-08-31 15:21:48,891] [    INFO][0m - loss: 0.60147681, learning_rate: 2.73e-06, global_step: 90, interval_runtime: 0.9499, interval_samples_per_second: 8.422, interval_steps_per_second: 10.527, epoch: 4.5[0m
[32m[2022-08-31 15:21:49,786] [    INFO][0m - loss: 0.50334263, learning_rate: 2.7e-06, global_step: 100, interval_runtime: 0.8952, interval_samples_per_second: 8.937, interval_steps_per_second: 11.171, epoch: 5.0[0m
[32m[2022-08-31 15:21:49,787] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 15:21:49,787] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 15:21:49,787] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:21:49,787] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:21:49,787] [    INFO][0m -   Total prediction steps = 3[0m
[32m[2022-08-31 15:21:50,490] [    INFO][0m - eval_loss: 0.7092321515083313, eval_accuracy: 0.71875, eval_runtime: 0.702, eval_samples_per_second: 227.93, eval_steps_per_second: 4.274, epoch: 5.0[0m
[32m[2022-08-31 15:21:50,491] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-100[0m
[32m[2022-08-31 15:21:50,491] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:21:53,688] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-100/tokenizer_config.json[0m
[32m[2022-08-31 15:21:53,688] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-100/special_tokens_map.json[0m
[32m[2022-08-31 15:22:00,347] [    INFO][0m - loss: 0.62328148, learning_rate: 2.6700000000000003e-06, global_step: 110, interval_runtime: 10.5601, interval_samples_per_second: 0.758, interval_steps_per_second: 0.947, epoch: 5.5[0m
[32m[2022-08-31 15:22:01,231] [    INFO][0m - loss: 0.37664752, learning_rate: 2.64e-06, global_step: 120, interval_runtime: 0.8843, interval_samples_per_second: 9.047, interval_steps_per_second: 11.309, epoch: 6.0[0m
[32m[2022-08-31 15:22:02,177] [    INFO][0m - loss: 0.50325227, learning_rate: 2.61e-06, global_step: 130, interval_runtime: 0.9457, interval_samples_per_second: 8.46, interval_steps_per_second: 10.574, epoch: 6.5[0m
[32m[2022-08-31 15:22:03,058] [    INFO][0m - loss: 0.46647243, learning_rate: 2.58e-06, global_step: 140, interval_runtime: 0.8819, interval_samples_per_second: 9.071, interval_steps_per_second: 11.339, epoch: 7.0[0m
[32m[2022-08-31 15:22:04,006] [    INFO][0m - loss: 0.43043752, learning_rate: 2.55e-06, global_step: 150, interval_runtime: 0.948, interval_samples_per_second: 8.439, interval_steps_per_second: 10.548, epoch: 7.5[0m
[32m[2022-08-31 15:22:04,896] [    INFO][0m - loss: 0.39854906, learning_rate: 2.52e-06, global_step: 160, interval_runtime: 0.8896, interval_samples_per_second: 8.993, interval_steps_per_second: 11.241, epoch: 8.0[0m
[32m[2022-08-31 15:22:05,840] [    INFO][0m - loss: 0.41871018, learning_rate: 2.49e-06, global_step: 170, interval_runtime: 0.9437, interval_samples_per_second: 8.477, interval_steps_per_second: 10.596, epoch: 8.5[0m
[32m[2022-08-31 15:22:06,730] [    INFO][0m - loss: 0.40881653, learning_rate: 2.4599999999999997e-06, global_step: 180, interval_runtime: 0.8898, interval_samples_per_second: 8.991, interval_steps_per_second: 11.239, epoch: 9.0[0m
[32m[2022-08-31 15:22:07,666] [    INFO][0m - loss: 0.34469995, learning_rate: 2.43e-06, global_step: 190, interval_runtime: 0.9367, interval_samples_per_second: 8.54, interval_steps_per_second: 10.676, epoch: 9.5[0m
[32m[2022-08-31 15:22:08,552] [    INFO][0m - loss: 0.27687857, learning_rate: 2.4000000000000003e-06, global_step: 200, interval_runtime: 0.8858, interval_samples_per_second: 9.031, interval_steps_per_second: 11.289, epoch: 10.0[0m
[32m[2022-08-31 15:22:08,553] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 15:22:08,553] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 15:22:08,553] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:22:08,553] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:22:08,553] [    INFO][0m -   Total prediction steps = 3[0m
[32m[2022-08-31 15:22:09,255] [    INFO][0m - eval_loss: 0.7866717576980591, eval_accuracy: 0.725, eval_runtime: 0.7014, eval_samples_per_second: 228.107, eval_steps_per_second: 4.277, epoch: 10.0[0m
[32m[2022-08-31 15:22:09,256] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-200[0m
[32m[2022-08-31 15:22:09,256] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:22:12,495] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-200/tokenizer_config.json[0m
[32m[2022-08-31 15:22:12,495] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-200/special_tokens_map.json[0m
[32m[2022-08-31 15:22:18,846] [    INFO][0m - loss: 0.29957242, learning_rate: 2.37e-06, global_step: 210, interval_runtime: 10.2934, interval_samples_per_second: 0.777, interval_steps_per_second: 0.972, epoch: 10.5[0m
[32m[2022-08-31 15:22:19,731] [    INFO][0m - loss: 0.27141168, learning_rate: 2.34e-06, global_step: 220, interval_runtime: 0.8854, interval_samples_per_second: 9.036, interval_steps_per_second: 11.295, epoch: 11.0[0m
[32m[2022-08-31 15:22:20,676] [    INFO][0m - loss: 0.21918805, learning_rate: 2.31e-06, global_step: 230, interval_runtime: 0.9447, interval_samples_per_second: 8.468, interval_steps_per_second: 10.585, epoch: 11.5[0m
[32m[2022-08-31 15:22:21,561] [    INFO][0m - loss: 0.28518944, learning_rate: 2.28e-06, global_step: 240, interval_runtime: 0.8851, interval_samples_per_second: 9.039, interval_steps_per_second: 11.298, epoch: 12.0[0m
[32m[2022-08-31 15:22:22,508] [    INFO][0m - loss: 0.27030945, learning_rate: 2.25e-06, global_step: 250, interval_runtime: 0.9471, interval_samples_per_second: 8.446, interval_steps_per_second: 10.558, epoch: 12.5[0m
[32m[2022-08-31 15:22:23,391] [    INFO][0m - loss: 0.20221722, learning_rate: 2.22e-06, global_step: 260, interval_runtime: 0.8831, interval_samples_per_second: 9.059, interval_steps_per_second: 11.323, epoch: 13.0[0m
[32m[2022-08-31 15:22:24,349] [    INFO][0m - loss: 0.16675701, learning_rate: 2.19e-06, global_step: 270, interval_runtime: 0.9577, interval_samples_per_second: 8.354, interval_steps_per_second: 10.442, epoch: 13.5[0m
[32m[2022-08-31 15:22:25,230] [    INFO][0m - loss: 0.28760819, learning_rate: 2.16e-06, global_step: 280, interval_runtime: 0.8809, interval_samples_per_second: 9.081, interval_steps_per_second: 11.352, epoch: 14.0[0m
[32m[2022-08-31 15:22:26,198] [    INFO][0m - loss: 0.18897935, learning_rate: 2.13e-06, global_step: 290, interval_runtime: 0.9685, interval_samples_per_second: 8.26, interval_steps_per_second: 10.325, epoch: 14.5[0m
[32m[2022-08-31 15:22:27,087] [    INFO][0m - loss: 0.18198267, learning_rate: 2.1e-06, global_step: 300, interval_runtime: 0.8893, interval_samples_per_second: 8.996, interval_steps_per_second: 11.245, epoch: 15.0[0m
[32m[2022-08-31 15:22:27,088] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 15:22:27,088] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 15:22:27,088] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:22:27,088] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:22:27,088] [    INFO][0m -   Total prediction steps = 3[0m
[32m[2022-08-31 15:22:27,801] [    INFO][0m - eval_loss: 0.9473894238471985, eval_accuracy: 0.74375, eval_runtime: 0.7118, eval_samples_per_second: 224.788, eval_steps_per_second: 4.215, epoch: 15.0[0m
[32m[2022-08-31 15:22:27,801] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-300[0m
[32m[2022-08-31 15:22:27,801] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:22:31,258] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-300/tokenizer_config.json[0m
[32m[2022-08-31 15:22:31,259] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-300/special_tokens_map.json[0m
[32m[2022-08-31 15:22:37,881] [    INFO][0m - loss: 0.1451725, learning_rate: 2.07e-06, global_step: 310, interval_runtime: 10.7941, interval_samples_per_second: 0.741, interval_steps_per_second: 0.926, epoch: 15.5[0m
[32m[2022-08-31 15:22:38,763] [    INFO][0m - loss: 0.16307399, learning_rate: 2.0400000000000004e-06, global_step: 320, interval_runtime: 0.881, interval_samples_per_second: 9.08, interval_steps_per_second: 11.35, epoch: 16.0[0m
[32m[2022-08-31 15:22:39,710] [    INFO][0m - loss: 0.19445801, learning_rate: 2.0100000000000002e-06, global_step: 330, interval_runtime: 0.947, interval_samples_per_second: 8.448, interval_steps_per_second: 10.56, epoch: 16.5[0m
[32m[2022-08-31 15:22:40,591] [    INFO][0m - loss: 0.14255142, learning_rate: 1.98e-06, global_step: 340, interval_runtime: 0.8813, interval_samples_per_second: 9.077, interval_steps_per_second: 11.347, epoch: 17.0[0m
[32m[2022-08-31 15:22:41,525] [    INFO][0m - loss: 0.10165586, learning_rate: 1.95e-06, global_step: 350, interval_runtime: 0.9337, interval_samples_per_second: 8.568, interval_steps_per_second: 10.71, epoch: 17.5[0m
[32m[2022-08-31 15:22:42,421] [    INFO][0m - loss: 0.18817208, learning_rate: 1.9200000000000003e-06, global_step: 360, interval_runtime: 0.8961, interval_samples_per_second: 8.927, interval_steps_per_second: 11.159, epoch: 18.0[0m
[32m[2022-08-31 15:22:43,360] [    INFO][0m - loss: 0.15301428, learning_rate: 1.8900000000000001e-06, global_step: 370, interval_runtime: 0.939, interval_samples_per_second: 8.52, interval_steps_per_second: 10.65, epoch: 18.5[0m
[32m[2022-08-31 15:22:44,245] [    INFO][0m - loss: 0.11218483, learning_rate: 1.86e-06, global_step: 380, interval_runtime: 0.8851, interval_samples_per_second: 9.039, interval_steps_per_second: 11.299, epoch: 19.0[0m
[32m[2022-08-31 15:22:45,187] [    INFO][0m - loss: 0.06697962, learning_rate: 1.83e-06, global_step: 390, interval_runtime: 0.9422, interval_samples_per_second: 8.491, interval_steps_per_second: 10.614, epoch: 19.5[0m
[32m[2022-08-31 15:22:46,071] [    INFO][0m - loss: 0.12943896, learning_rate: 1.8e-06, global_step: 400, interval_runtime: 0.8838, interval_samples_per_second: 9.051, interval_steps_per_second: 11.314, epoch: 20.0[0m
[32m[2022-08-31 15:22:46,072] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 15:22:46,072] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 15:22:46,072] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:22:46,072] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:22:46,072] [    INFO][0m -   Total prediction steps = 3[0m
[32m[2022-08-31 15:22:46,785] [    INFO][0m - eval_loss: 1.186769723892212, eval_accuracy: 0.74375, eval_runtime: 0.7129, eval_samples_per_second: 224.421, eval_steps_per_second: 4.208, epoch: 20.0[0m
[32m[2022-08-31 15:22:46,785] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-400[0m
[32m[2022-08-31 15:22:46,786] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:22:49,943] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-400/tokenizer_config.json[0m
[32m[2022-08-31 15:22:49,944] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-400/special_tokens_map.json[0m
[32m[2022-08-31 15:22:56,024] [    INFO][0m - loss: 0.11468244, learning_rate: 1.77e-06, global_step: 410, interval_runtime: 9.9533, interval_samples_per_second: 0.804, interval_steps_per_second: 1.005, epoch: 20.5[0m
[32m[2022-08-31 15:22:56,919] [    INFO][0m - loss: 0.0772739, learning_rate: 1.7399999999999999e-06, global_step: 420, interval_runtime: 0.8945, interval_samples_per_second: 8.944, interval_steps_per_second: 11.18, epoch: 21.0[0m
[32m[2022-08-31 15:22:57,857] [    INFO][0m - loss: 0.09580462, learning_rate: 1.71e-06, global_step: 430, interval_runtime: 0.9382, interval_samples_per_second: 8.527, interval_steps_per_second: 10.659, epoch: 21.5[0m
[32m[2022-08-31 15:22:58,749] [    INFO][0m - loss: 0.16464843, learning_rate: 1.6800000000000002e-06, global_step: 440, interval_runtime: 0.8923, interval_samples_per_second: 8.965, interval_steps_per_second: 11.207, epoch: 22.0[0m
[32m[2022-08-31 15:22:59,695] [    INFO][0m - loss: 0.08541673, learning_rate: 1.65e-06, global_step: 450, interval_runtime: 0.9456, interval_samples_per_second: 8.46, interval_steps_per_second: 10.575, epoch: 22.5[0m
[32m[2022-08-31 15:23:00,604] [    INFO][0m - loss: 0.08581613, learning_rate: 1.6200000000000002e-06, global_step: 460, interval_runtime: 0.9097, interval_samples_per_second: 8.794, interval_steps_per_second: 10.992, epoch: 23.0[0m
[32m[2022-08-31 15:23:01,555] [    INFO][0m - loss: 0.0787383, learning_rate: 1.59e-06, global_step: 470, interval_runtime: 0.9509, interval_samples_per_second: 8.413, interval_steps_per_second: 10.517, epoch: 23.5[0m
[32m[2022-08-31 15:23:02,440] [    INFO][0m - loss: 0.08566085, learning_rate: 1.56e-06, global_step: 480, interval_runtime: 0.8847, interval_samples_per_second: 9.043, interval_steps_per_second: 11.303, epoch: 24.0[0m
[32m[2022-08-31 15:23:03,386] [    INFO][0m - loss: 0.07923208, learning_rate: 1.53e-06, global_step: 490, interval_runtime: 0.9459, interval_samples_per_second: 8.457, interval_steps_per_second: 10.572, epoch: 24.5[0m
[32m[2022-08-31 15:23:04,272] [    INFO][0m - loss: 0.07441272, learning_rate: 1.5e-06, global_step: 500, interval_runtime: 0.8859, interval_samples_per_second: 9.03, interval_steps_per_second: 11.288, epoch: 25.0[0m
[32m[2022-08-31 15:23:04,272] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 15:23:04,273] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 15:23:04,273] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:23:04,273] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:23:04,273] [    INFO][0m -   Total prediction steps = 3[0m
[32m[2022-08-31 15:23:04,970] [    INFO][0m - eval_loss: 1.4473134279251099, eval_accuracy: 0.73125, eval_runtime: 0.6968, eval_samples_per_second: 229.637, eval_steps_per_second: 4.306, epoch: 25.0[0m
[32m[2022-08-31 15:23:04,970] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-500[0m
[32m[2022-08-31 15:23:04,970] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:23:08,170] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-500/tokenizer_config.json[0m
[32m[2022-08-31 15:23:08,170] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-500/special_tokens_map.json[0m
[32m[2022-08-31 15:23:14,368] [    INFO][0m - loss: 0.0456175, learning_rate: 1.4700000000000001e-06, global_step: 510, interval_runtime: 10.096, interval_samples_per_second: 0.792, interval_steps_per_second: 0.99, epoch: 25.5[0m
[32m[2022-08-31 15:23:15,259] [    INFO][0m - loss: 0.07844511, learning_rate: 1.44e-06, global_step: 520, interval_runtime: 0.8914, interval_samples_per_second: 8.975, interval_steps_per_second: 11.218, epoch: 26.0[0m
[32m[2022-08-31 15:23:16,213] [    INFO][0m - loss: 0.01945523, learning_rate: 1.41e-06, global_step: 530, interval_runtime: 0.9536, interval_samples_per_second: 8.39, interval_steps_per_second: 10.487, epoch: 26.5[0m
[32m[2022-08-31 15:23:17,103] [    INFO][0m - loss: 0.09362832, learning_rate: 1.3800000000000001e-06, global_step: 540, interval_runtime: 0.8894, interval_samples_per_second: 8.995, interval_steps_per_second: 11.243, epoch: 27.0[0m
[32m[2022-08-31 15:23:18,045] [    INFO][0m - loss: 0.02446018, learning_rate: 1.35e-06, global_step: 550, interval_runtime: 0.9424, interval_samples_per_second: 8.489, interval_steps_per_second: 10.611, epoch: 27.5[0m
[32m[2022-08-31 15:23:18,933] [    INFO][0m - loss: 0.08990087, learning_rate: 1.32e-06, global_step: 560, interval_runtime: 0.8881, interval_samples_per_second: 9.008, interval_steps_per_second: 11.26, epoch: 28.0[0m
[32m[2022-08-31 15:23:19,879] [    INFO][0m - loss: 0.03099898, learning_rate: 1.29e-06, global_step: 570, interval_runtime: 0.9463, interval_samples_per_second: 8.454, interval_steps_per_second: 10.568, epoch: 28.5[0m
[32m[2022-08-31 15:23:20,770] [    INFO][0m - loss: 0.09503048, learning_rate: 1.26e-06, global_step: 580, interval_runtime: 0.8907, interval_samples_per_second: 8.982, interval_steps_per_second: 11.227, epoch: 29.0[0m
[32m[2022-08-31 15:23:21,710] [    INFO][0m - loss: 0.02651134, learning_rate: 1.2299999999999999e-06, global_step: 590, interval_runtime: 0.9403, interval_samples_per_second: 8.508, interval_steps_per_second: 10.635, epoch: 29.5[0m
[32m[2022-08-31 15:23:22,600] [    INFO][0m - loss: 0.04969427, learning_rate: 1.2000000000000002e-06, global_step: 600, interval_runtime: 0.8895, interval_samples_per_second: 8.994, interval_steps_per_second: 11.242, epoch: 30.0[0m
[32m[2022-08-31 15:23:22,600] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 15:23:22,600] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 15:23:22,600] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:23:22,600] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:23:22,601] [    INFO][0m -   Total prediction steps = 3[0m
[32m[2022-08-31 15:23:23,322] [    INFO][0m - eval_loss: 1.645745873451233, eval_accuracy: 0.73125, eval_runtime: 0.7211, eval_samples_per_second: 221.896, eval_steps_per_second: 4.161, epoch: 30.0[0m
[32m[2022-08-31 15:23:23,322] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-600[0m
[32m[2022-08-31 15:23:23,322] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:23:26,833] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-600/tokenizer_config.json[0m
[32m[2022-08-31 15:23:26,834] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-600/special_tokens_map.json[0m
[32m[2022-08-31 15:23:33,166] [    INFO][0m - loss: 0.03222713, learning_rate: 1.17e-06, global_step: 610, interval_runtime: 10.5664, interval_samples_per_second: 0.757, interval_steps_per_second: 0.946, epoch: 30.5[0m
[32m[2022-08-31 15:23:34,051] [    INFO][0m - loss: 0.02648944, learning_rate: 1.14e-06, global_step: 620, interval_runtime: 0.8852, interval_samples_per_second: 9.037, interval_steps_per_second: 11.297, epoch: 31.0[0m
[32m[2022-08-31 15:23:34,991] [    INFO][0m - loss: 0.04492521, learning_rate: 1.11e-06, global_step: 630, interval_runtime: 0.9392, interval_samples_per_second: 8.518, interval_steps_per_second: 10.648, epoch: 31.5[0m
[32m[2022-08-31 15:23:35,877] [    INFO][0m - loss: 0.02892792, learning_rate: 1.08e-06, global_step: 640, interval_runtime: 0.8863, interval_samples_per_second: 9.027, interval_steps_per_second: 11.283, epoch: 32.0[0m
[32m[2022-08-31 15:23:36,816] [    INFO][0m - loss: 0.01329842, learning_rate: 1.05e-06, global_step: 650, interval_runtime: 0.9393, interval_samples_per_second: 8.517, interval_steps_per_second: 10.647, epoch: 32.5[0m
[32m[2022-08-31 15:23:37,704] [    INFO][0m - loss: 0.035778, learning_rate: 1.0200000000000002e-06, global_step: 660, interval_runtime: 0.8874, interval_samples_per_second: 9.015, interval_steps_per_second: 11.269, epoch: 33.0[0m
[32m[2022-08-31 15:23:38,659] [    INFO][0m - loss: 0.02285399, learning_rate: 9.9e-07, global_step: 670, interval_runtime: 0.9553, interval_samples_per_second: 8.374, interval_steps_per_second: 10.468, epoch: 33.5[0m
[32m[2022-08-31 15:23:39,544] [    INFO][0m - loss: 0.03410998, learning_rate: 9.600000000000001e-07, global_step: 680, interval_runtime: 0.8851, interval_samples_per_second: 9.039, interval_steps_per_second: 11.298, epoch: 34.0[0m
[32m[2022-08-31 15:23:40,512] [    INFO][0m - loss: 0.06413728, learning_rate: 9.3e-07, global_step: 690, interval_runtime: 0.9681, interval_samples_per_second: 8.263, interval_steps_per_second: 10.329, epoch: 34.5[0m
[32m[2022-08-31 15:23:41,420] [    INFO][0m - loss: 0.02603287, learning_rate: 9e-07, global_step: 700, interval_runtime: 0.9081, interval_samples_per_second: 8.809, interval_steps_per_second: 11.012, epoch: 35.0[0m
[32m[2022-08-31 15:23:41,421] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 15:23:41,421] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 15:23:41,421] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:23:41,421] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:23:41,422] [    INFO][0m -   Total prediction steps = 3[0m
[32m[2022-08-31 15:23:42,146] [    INFO][0m - eval_loss: 1.8338886499404907, eval_accuracy: 0.74375, eval_runtime: 0.7245, eval_samples_per_second: 220.828, eval_steps_per_second: 4.141, epoch: 35.0[0m
[32m[2022-08-31 15:23:42,147] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-700[0m
[32m[2022-08-31 15:23:42,147] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:23:46,006] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-700/tokenizer_config.json[0m
[32m[2022-08-31 15:23:46,007] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-700/special_tokens_map.json[0m
[32m[2022-08-31 15:23:51,560] [    INFO][0m - 
Training completed. 
[0m
[32m[2022-08-31 15:23:51,561] [    INFO][0m - Loading best model from ./checkpoints/checkpoint-300 (score: 0.74375).[0m
[32m[2022-08-31 15:23:52,798] [    INFO][0m - train_runtime: 133.3765, train_samples_per_second: 59.981, train_steps_per_second: 7.498, train_loss: 0.2322504424410207, epoch: 35.0[0m
[32m[2022-08-31 15:23:52,800] [    INFO][0m - Saving model checkpoint to ./checkpoints/[0m
[32m[2022-08-31 15:23:52,800] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:23:55,909] [    INFO][0m - tokenizer config file saved in ./checkpoints/tokenizer_config.json[0m
[32m[2022-08-31 15:23:55,910] [    INFO][0m - Special tokens file saved in ./checkpoints/special_tokens_map.json[0m
[32m[2022-08-31 15:23:55,912] [    INFO][0m - ***** train metrics *****[0m
[32m[2022-08-31 15:23:55,912] [    INFO][0m -   epoch                    =       35.0[0m
[32m[2022-08-31 15:23:55,912] [    INFO][0m -   train_loss               =     0.2323[0m
[32m[2022-08-31 15:23:55,912] [    INFO][0m -   train_runtime            = 0:02:13.37[0m
[32m[2022-08-31 15:23:55,912] [    INFO][0m -   train_samples_per_second =     59.981[0m
[32m[2022-08-31 15:23:55,912] [    INFO][0m -   train_steps_per_second   =      7.498[0m
[32m[2022-08-31 15:23:55,916] [    INFO][0m - ***** Running Prediction *****[0m
[32m[2022-08-31 15:23:55,916] [    INFO][0m -   Num examples = 2520[0m
[32m[2022-08-31 15:23:55,916] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:23:55,917] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:23:55,917] [    INFO][0m -   Total prediction steps = 40[0m
[32m[2022-08-31 15:24:06,922] [    INFO][0m - ***** test metrics *****[0m
[32m[2022-08-31 15:24:06,922] [    INFO][0m -   test_accuracy           =     0.7163[0m
[32m[2022-08-31 15:24:06,923] [    INFO][0m -   test_loss               =     0.9161[0m
[32m[2022-08-31 15:24:06,923] [    INFO][0m -   test_runtime            = 0:00:11.00[0m
[32m[2022-08-31 15:24:06,923] [    INFO][0m -   test_samples_per_second =    228.979[0m
[32m[2022-08-31 15:24:06,923] [    INFO][0m -   test_steps_per_second   =      3.635[0m
[32m[2022-08-31 15:24:06,923] [    INFO][0m - ***** Running Prediction *****[0m
[32m[2022-08-31 15:24:06,924] [    INFO][0m -   Num examples = 3000[0m
[32m[2022-08-31 15:24:06,924] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:24:06,924] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:24:06,924] [    INFO][0m -   Total prediction steps = 47[0m
[32m[2022-08-31 15:24:23,016] [    INFO][0m - Predictions for ocnlif saved to ./fewclue_submit_examples.[0m
[]
run.sh: line 71: --freeze_plm: command not found
