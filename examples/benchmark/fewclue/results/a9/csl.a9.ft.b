[33m[2022-08-31 16:12:20,517] [ WARNING][0m - evaluation_strategy reset to IntervalStrategy.STEPS for do_eval is True. you can also set evaluation_strategy='epoch'.[0m
[32m[2022-08-31 16:12:20,517] [    INFO][0m - The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).[0m
[32m[2022-08-31 16:12:20,517] [    INFO][0m - ============================================================[0m
[32m[2022-08-31 16:12:20,517] [    INFO][0m -      Model Configuration Arguments      [0m
[32m[2022-08-31 16:12:20,517] [    INFO][0m - paddle commit id              :65f388690048d0965ec7d3b43fb6ee9d8c6dee7c[0m
[32m[2022-08-31 16:12:20,517] [    INFO][0m - do_save                       :True[0m
[32m[2022-08-31 16:12:20,517] [    INFO][0m - do_test                       :True[0m
[32m[2022-08-31 16:12:20,518] [    INFO][0m - early_stop_patience           :4[0m
[32m[2022-08-31 16:12:20,518] [    INFO][0m - export_type                   :paddle[0m
[32m[2022-08-31 16:12:20,518] [    INFO][0m - model_name_or_path            :ernie-3.0-base-zh[0m
[32m[2022-08-31 16:12:20,518] [    INFO][0m - [0m
[32m[2022-08-31 16:12:20,518] [    INFO][0m - ============================================================[0m
[32m[2022-08-31 16:12:20,518] [    INFO][0m -       Data Configuration Arguments      [0m
[32m[2022-08-31 16:12:20,518] [    INFO][0m - paddle commit id              :65f388690048d0965ec7d3b43fb6ee9d8c6dee7c[0m
[32m[2022-08-31 16:12:20,518] [    INFO][0m - encoder_hidden_size           :200[0m
[32m[2022-08-31 16:12:20,518] [    INFO][0m - prompt                        :“{'text':'text_a'}”和“{'text':'text_b'}”之间的逻辑关系是{'mask'}{'mask'}。[0m
[32m[2022-08-31 16:12:20,518] [    INFO][0m - soft_encoder                  :lstm[0m
[32m[2022-08-31 16:12:20,518] [    INFO][0m - split_id                      :few_all[0m
[32m[2022-08-31 16:12:20,518] [    INFO][0m - task_name                     :csl[0m
[32m[2022-08-31 16:12:20,518] [    INFO][0m - [0m
[32m[2022-08-31 16:12:20,519] [    INFO][0m - Already cached /ssd2/wanghuijuan03/.paddlenlp/models/ernie-3.0-base-zh/ernie_3.0_base_zh.pdparams[0m
W0831 16:12:20.520505 49181 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 11.2
W0831 16:12:20.524647 49181 gpu_resources.cc:91] device: 0, cuDNN Version: 8.1.
[32m[2022-08-31 16:12:23,626] [    INFO][0m - Already cached /ssd2/wanghuijuan03/.paddlenlp/models/ernie-3.0-base-zh/ernie_3.0_base_zh_vocab.txt[0m
[32m[2022-08-31 16:12:23,651] [    INFO][0m - tokenizer config file saved in /ssd2/wanghuijuan03/.paddlenlp/models/ernie-3.0-base-zh/tokenizer_config.json[0m
[32m[2022-08-31 16:12:23,651] [    INFO][0m - Special tokens file saved in /ssd2/wanghuijuan03/.paddlenlp/models/ernie-3.0-base-zh/special_tokens_map.json[0m
[32m[2022-08-31 16:12:23,653] [    INFO][0m - Using template: [{'add_prefix_space': '', 'hard': '“'}, {'add_prefix_space': '', 'text': 'text_a'}, {'add_prefix_space': '', 'hard': '”和“'}, {'add_prefix_space': '', 'text': 'text_b'}, {'add_prefix_space': '', 'hard': '”之间的逻辑关系是'}, {'add_prefix_space': '', 'mask': None}, {'add_prefix_space': '', 'mask': None}, {'add_prefix_space': '', 'hard': '。'}][0m
[32m[2022-08-31 16:12:23,655] [    INFO][0m - {'0': 0, '1': 1}[0m
2022-08-31 16:12:23,657 INFO [download.py:119] unique_endpoints {''}
[32m[2022-08-31 16:12:24,956] [    INFO][0m - ============================================================[0m
[32m[2022-08-31 16:12:24,956] [    INFO][0m -     Training Configuration Arguments    [0m
[32m[2022-08-31 16:12:24,956] [    INFO][0m - paddle commit id              :65f388690048d0965ec7d3b43fb6ee9d8c6dee7c[0m
[32m[2022-08-31 16:12:24,956] [    INFO][0m - _no_sync_in_gradient_accumulation:True[0m
[32m[2022-08-31 16:12:24,956] [    INFO][0m - adam_beta1                    :0.9[0m
[32m[2022-08-31 16:12:24,956] [    INFO][0m - adam_beta2                    :0.999[0m
[32m[2022-08-31 16:12:24,956] [    INFO][0m - adam_epsilon                  :1e-08[0m
[32m[2022-08-31 16:12:24,956] [    INFO][0m - alpha_rdrop                   :5.0[0m
[32m[2022-08-31 16:12:24,956] [    INFO][0m - alpha_rgl                     :0.5[0m
[32m[2022-08-31 16:12:24,957] [    INFO][0m - current_device                :gpu:0[0m
[32m[2022-08-31 16:12:24,957] [    INFO][0m - dataloader_drop_last          :False[0m
[32m[2022-08-31 16:12:24,957] [    INFO][0m - dataloader_num_workers        :0[0m
[32m[2022-08-31 16:12:24,957] [    INFO][0m - device                        :gpu[0m
[32m[2022-08-31 16:12:24,957] [    INFO][0m - disable_tqdm                  :True[0m
[32m[2022-08-31 16:12:24,957] [    INFO][0m - do_eval                       :True[0m
[32m[2022-08-31 16:12:24,957] [    INFO][0m - do_export                     :False[0m
[32m[2022-08-31 16:12:24,957] [    INFO][0m - do_predict                    :True[0m
[32m[2022-08-31 16:12:24,957] [    INFO][0m - do_train                      :True[0m
[32m[2022-08-31 16:12:24,957] [    INFO][0m - eval_batch_size               :32[0m
[32m[2022-08-31 16:12:24,957] [    INFO][0m - eval_steps                    :100[0m
[32m[2022-08-31 16:12:24,957] [    INFO][0m - evaluation_strategy           :IntervalStrategy.STEPS[0m
[32m[2022-08-31 16:12:24,958] [    INFO][0m - first_max_length              :None[0m
[32m[2022-08-31 16:12:24,958] [    INFO][0m - fp16                          :False[0m
[32m[2022-08-31 16:12:24,958] [    INFO][0m - fp16_opt_level                :O1[0m
[32m[2022-08-31 16:12:24,958] [    INFO][0m - freeze_dropout                :False[0m
[32m[2022-08-31 16:12:24,958] [    INFO][0m - freeze_plm                    :False[0m
[32m[2022-08-31 16:12:24,958] [    INFO][0m - gradient_accumulation_steps   :1[0m
[32m[2022-08-31 16:12:24,958] [    INFO][0m - greater_is_better             :True[0m
[32m[2022-08-31 16:12:24,958] [    INFO][0m - ignore_data_skip              :False[0m
[32m[2022-08-31 16:12:24,958] [    INFO][0m - label_names                   :None[0m
[32m[2022-08-31 16:12:24,958] [    INFO][0m - learning_rate                 :3e-05[0m
[32m[2022-08-31 16:12:24,958] [    INFO][0m - load_best_model_at_end        :True[0m
[32m[2022-08-31 16:12:24,958] [    INFO][0m - local_process_index           :0[0m
[32m[2022-08-31 16:12:24,958] [    INFO][0m - local_rank                    :-1[0m
[32m[2022-08-31 16:12:24,958] [    INFO][0m - log_level                     :-1[0m
[32m[2022-08-31 16:12:24,959] [    INFO][0m - log_level_replica             :-1[0m
[32m[2022-08-31 16:12:24,959] [    INFO][0m - log_on_each_node              :True[0m
[32m[2022-08-31 16:12:24,959] [    INFO][0m - logging_dir                   :./checkpoints/runs/Aug31_16-12-20_instance-3bwob41y-01[0m
[32m[2022-08-31 16:12:24,959] [    INFO][0m - logging_first_step            :False[0m
[32m[2022-08-31 16:12:24,959] [    INFO][0m - logging_steps                 :10[0m
[32m[2022-08-31 16:12:24,959] [    INFO][0m - logging_strategy              :IntervalStrategy.STEPS[0m
[32m[2022-08-31 16:12:24,959] [    INFO][0m - lr_scheduler_type             :SchedulerType.LINEAR[0m
[32m[2022-08-31 16:12:24,959] [    INFO][0m - max_grad_norm                 :1.0[0m
[32m[2022-08-31 16:12:24,959] [    INFO][0m - max_seq_length                :320[0m
[32m[2022-08-31 16:12:24,959] [    INFO][0m - max_steps                     :-1[0m
[32m[2022-08-31 16:12:24,959] [    INFO][0m - metric_for_best_model         :accuracy[0m
[32m[2022-08-31 16:12:24,959] [    INFO][0m - minimum_eval_times            :None[0m
[32m[2022-08-31 16:12:24,959] [    INFO][0m - no_cuda                       :False[0m
[32m[2022-08-31 16:12:24,960] [    INFO][0m - num_train_epochs              :50.0[0m
[32m[2022-08-31 16:12:24,960] [    INFO][0m - optim                         :OptimizerNames.ADAMW[0m
[32m[2022-08-31 16:12:24,960] [    INFO][0m - other_max_length              :None[0m
[32m[2022-08-31 16:12:24,960] [    INFO][0m - output_dir                    :./checkpoints/[0m
[32m[2022-08-31 16:12:24,960] [    INFO][0m - overwrite_output_dir          :False[0m
[32m[2022-08-31 16:12:24,960] [    INFO][0m - past_index                    :-1[0m
[32m[2022-08-31 16:12:24,960] [    INFO][0m - per_device_eval_batch_size    :32[0m
[32m[2022-08-31 16:12:24,960] [    INFO][0m - per_device_train_batch_size   :8[0m
[32m[2022-08-31 16:12:24,960] [    INFO][0m - ppt_adam_beta1                :0.9[0m
[32m[2022-08-31 16:12:24,960] [    INFO][0m - ppt_adam_beta2                :0.999[0m
[32m[2022-08-31 16:12:24,960] [    INFO][0m - ppt_adam_epsilon              :1e-08[0m
[32m[2022-08-31 16:12:24,960] [    INFO][0m - ppt_learning_rate             :0.0003[0m
[32m[2022-08-31 16:12:24,960] [    INFO][0m - ppt_weight_decay              :0.0[0m
[32m[2022-08-31 16:12:24,960] [    INFO][0m - prediction_loss_only          :False[0m
[32m[2022-08-31 16:12:24,961] [    INFO][0m - process_index                 :0[0m
[32m[2022-08-31 16:12:24,961] [    INFO][0m - remove_unused_columns         :True[0m
[32m[2022-08-31 16:12:24,961] [    INFO][0m - report_to                     :['visualdl'][0m
[32m[2022-08-31 16:12:24,961] [    INFO][0m - resume_from_checkpoint        :None[0m
[32m[2022-08-31 16:12:24,961] [    INFO][0m - run_name                      :./checkpoints/[0m
[32m[2022-08-31 16:12:24,961] [    INFO][0m - save_on_each_node             :False[0m
[32m[2022-08-31 16:12:24,961] [    INFO][0m - save_steps                    :100[0m
[32m[2022-08-31 16:12:24,961] [    INFO][0m - save_strategy                 :IntervalStrategy.STEPS[0m
[32m[2022-08-31 16:12:24,961] [    INFO][0m - save_total_limit              :None[0m
[32m[2022-08-31 16:12:24,961] [    INFO][0m - scale_loss                    :32768[0m
[32m[2022-08-31 16:12:24,961] [    INFO][0m - seed                          :42[0m
[32m[2022-08-31 16:12:24,961] [    INFO][0m - should_log                    :True[0m
[32m[2022-08-31 16:12:24,961] [    INFO][0m - should_save                   :True[0m
[32m[2022-08-31 16:12:24,962] [    INFO][0m - task_type                     :multi-class[0m
[32m[2022-08-31 16:12:24,962] [    INFO][0m - train_batch_size              :8[0m
[32m[2022-08-31 16:12:24,962] [    INFO][0m - truncate_mode                 :tail[0m
[32m[2022-08-31 16:12:24,962] [    INFO][0m - use_rdrop                     :False[0m
[32m[2022-08-31 16:12:24,962] [    INFO][0m - use_rgl                       :False[0m
[32m[2022-08-31 16:12:24,962] [    INFO][0m - warmup_ratio                  :0.0[0m
[32m[2022-08-31 16:12:24,962] [    INFO][0m - warmup_steps                  :0[0m
[32m[2022-08-31 16:12:24,962] [    INFO][0m - weight_decay                  :0.0[0m
[32m[2022-08-31 16:12:24,962] [    INFO][0m - world_size                    :1[0m
[32m[2022-08-31 16:12:24,962] [    INFO][0m - [0m
[32m[2022-08-31 16:12:24,964] [    INFO][0m - ***** Running training *****[0m
[32m[2022-08-31 16:12:24,964] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 16:12:24,964] [    INFO][0m -   Num Epochs = 50[0m
[32m[2022-08-31 16:12:24,964] [    INFO][0m -   Instantaneous batch size per device = 8[0m
[32m[2022-08-31 16:12:24,964] [    INFO][0m -   Total train batch size (w. parallel, distributed & accumulation) = 8[0m
[32m[2022-08-31 16:12:24,964] [    INFO][0m -   Gradient Accumulation steps = 1[0m
[32m[2022-08-31 16:12:24,964] [    INFO][0m -   Total optimization steps = 1000.0[0m
[32m[2022-08-31 16:12:24,964] [    INFO][0m -   Total num train samples = 8000[0m
[32m[2022-08-31 16:12:28,082] [    INFO][0m - loss: 0.86393633, learning_rate: 2.97e-05, global_step: 10, interval_runtime: 3.1165, interval_samples_per_second: 2.567, interval_steps_per_second: 3.209, epoch: 0.5[0m
[32m[2022-08-31 16:12:30,008] [    INFO][0m - loss: 0.68107295, learning_rate: 2.94e-05, global_step: 20, interval_runtime: 1.9257, interval_samples_per_second: 4.154, interval_steps_per_second: 5.193, epoch: 1.0[0m
[32m[2022-08-31 16:12:32,063] [    INFO][0m - loss: 0.65574307, learning_rate: 2.91e-05, global_step: 30, interval_runtime: 2.0556, interval_samples_per_second: 3.892, interval_steps_per_second: 4.865, epoch: 1.5[0m
[32m[2022-08-31 16:12:33,985] [    INFO][0m - loss: 0.67404661, learning_rate: 2.88e-05, global_step: 40, interval_runtime: 1.9216, interval_samples_per_second: 4.163, interval_steps_per_second: 5.204, epoch: 2.0[0m
[32m[2022-08-31 16:12:36,035] [    INFO][0m - loss: 0.6173296, learning_rate: 2.8499999999999998e-05, global_step: 50, interval_runtime: 2.0499, interval_samples_per_second: 3.903, interval_steps_per_second: 4.878, epoch: 2.5[0m
[32m[2022-08-31 16:12:37,961] [    INFO][0m - loss: 0.48679628, learning_rate: 2.8199999999999998e-05, global_step: 60, interval_runtime: 1.9266, interval_samples_per_second: 4.152, interval_steps_per_second: 5.19, epoch: 3.0[0m
[32m[2022-08-31 16:12:40,081] [    INFO][0m - loss: 0.31281869, learning_rate: 2.79e-05, global_step: 70, interval_runtime: 2.1197, interval_samples_per_second: 3.774, interval_steps_per_second: 4.718, epoch: 3.5[0m
[32m[2022-08-31 16:12:42,007] [    INFO][0m - loss: 0.22641175, learning_rate: 2.7600000000000003e-05, global_step: 80, interval_runtime: 1.9265, interval_samples_per_second: 4.153, interval_steps_per_second: 5.191, epoch: 4.0[0m
[32m[2022-08-31 16:12:44,056] [    INFO][0m - loss: 0.15334914, learning_rate: 2.7300000000000003e-05, global_step: 90, interval_runtime: 2.0491, interval_samples_per_second: 3.904, interval_steps_per_second: 4.88, epoch: 4.5[0m
[32m[2022-08-31 16:12:45,990] [    INFO][0m - loss: 0.12578151, learning_rate: 2.7000000000000002e-05, global_step: 100, interval_runtime: 1.9337, interval_samples_per_second: 4.137, interval_steps_per_second: 5.171, epoch: 5.0[0m
[32m[2022-08-31 16:12:45,991] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 16:12:45,991] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 16:12:45,991] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 16:12:45,991] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 16:12:45,991] [    INFO][0m -   Total prediction steps = 5[0m
[32m[2022-08-31 16:12:47,694] [    INFO][0m - eval_loss: 4.162437438964844, eval_accuracy: 0.54375, eval_runtime: 1.7027, eval_samples_per_second: 93.969, eval_steps_per_second: 2.937, epoch: 5.0[0m
[32m[2022-08-31 16:12:47,694] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-100[0m
[32m[2022-08-31 16:12:47,695] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 16:12:51,247] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-100/tokenizer_config.json[0m
[32m[2022-08-31 16:12:51,248] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-100/special_tokens_map.json[0m
[32m[2022-08-31 16:12:58,968] [    INFO][0m - loss: 0.25518095, learning_rate: 2.6700000000000002e-05, global_step: 110, interval_runtime: 12.9782, interval_samples_per_second: 0.616, interval_steps_per_second: 0.771, epoch: 5.5[0m
[32m[2022-08-31 16:13:00,900] [    INFO][0m - loss: 0.45808253, learning_rate: 2.64e-05, global_step: 120, interval_runtime: 1.9318, interval_samples_per_second: 4.141, interval_steps_per_second: 5.177, epoch: 6.0[0m
[32m[2022-08-31 16:13:02,943] [    INFO][0m - loss: 0.05229799, learning_rate: 2.61e-05, global_step: 130, interval_runtime: 2.0422, interval_samples_per_second: 3.917, interval_steps_per_second: 4.897, epoch: 6.5[0m
[32m[2022-08-31 16:13:04,879] [    INFO][0m - loss: 0.3509902, learning_rate: 2.58e-05, global_step: 140, interval_runtime: 1.9367, interval_samples_per_second: 4.131, interval_steps_per_second: 5.164, epoch: 7.0[0m
[32m[2022-08-31 16:13:06,956] [    INFO][0m - loss: 0.08966898, learning_rate: 2.55e-05, global_step: 150, interval_runtime: 2.0764, interval_samples_per_second: 3.853, interval_steps_per_second: 4.816, epoch: 7.5[0m
[32m[2022-08-31 16:13:08,891] [    INFO][0m - loss: 0.02781133, learning_rate: 2.52e-05, global_step: 160, interval_runtime: 1.9352, interval_samples_per_second: 4.134, interval_steps_per_second: 5.167, epoch: 8.0[0m
[32m[2022-08-31 16:13:10,962] [    INFO][0m - loss: 0.06338734, learning_rate: 2.49e-05, global_step: 170, interval_runtime: 2.0711, interval_samples_per_second: 3.863, interval_steps_per_second: 4.828, epoch: 8.5[0m
[32m[2022-08-31 16:13:12,891] [    INFO][0m - loss: 0.25021205, learning_rate: 2.4599999999999998e-05, global_step: 180, interval_runtime: 1.9293, interval_samples_per_second: 4.147, interval_steps_per_second: 5.183, epoch: 9.0[0m
[32m[2022-08-31 16:13:15,018] [    INFO][0m - loss: 0.2276804, learning_rate: 2.43e-05, global_step: 190, interval_runtime: 2.1266, interval_samples_per_second: 3.762, interval_steps_per_second: 4.702, epoch: 9.5[0m
[32m[2022-08-31 16:13:16,950] [    INFO][0m - loss: 0.0456407, learning_rate: 2.4e-05, global_step: 200, interval_runtime: 1.9323, interval_samples_per_second: 4.14, interval_steps_per_second: 5.175, epoch: 10.0[0m
[32m[2022-08-31 16:13:16,951] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 16:13:16,951] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 16:13:16,951] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 16:13:16,951] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 16:13:16,951] [    INFO][0m -   Total prediction steps = 5[0m
[32m[2022-08-31 16:13:18,782] [    INFO][0m - eval_loss: 3.8831565380096436, eval_accuracy: 0.55625, eval_runtime: 1.8302, eval_samples_per_second: 87.422, eval_steps_per_second: 2.732, epoch: 10.0[0m
[32m[2022-08-31 16:13:18,782] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-200[0m
[32m[2022-08-31 16:13:18,782] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 16:13:22,166] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-200/tokenizer_config.json[0m
[32m[2022-08-31 16:13:22,166] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-200/special_tokens_map.json[0m
[32m[2022-08-31 16:13:29,497] [    INFO][0m - loss: 0.07431381, learning_rate: 2.37e-05, global_step: 210, interval_runtime: 12.5469, interval_samples_per_second: 0.638, interval_steps_per_second: 0.797, epoch: 10.5[0m
[32m[2022-08-31 16:13:31,430] [    INFO][0m - loss: 0.21320906, learning_rate: 2.3400000000000003e-05, global_step: 220, interval_runtime: 1.9324, interval_samples_per_second: 4.14, interval_steps_per_second: 5.175, epoch: 11.0[0m
[32m[2022-08-31 16:13:33,506] [    INFO][0m - loss: 0.05051149, learning_rate: 2.3100000000000002e-05, global_step: 230, interval_runtime: 2.0762, interval_samples_per_second: 3.853, interval_steps_per_second: 4.816, epoch: 11.5[0m
[32m[2022-08-31 16:13:35,449] [    INFO][0m - loss: 0.25673246, learning_rate: 2.2800000000000002e-05, global_step: 240, interval_runtime: 1.9423, interval_samples_per_second: 4.119, interval_steps_per_second: 5.148, epoch: 12.0[0m
[32m[2022-08-31 16:13:37,543] [    INFO][0m - loss: 0.01583877, learning_rate: 2.25e-05, global_step: 250, interval_runtime: 2.0943, interval_samples_per_second: 3.82, interval_steps_per_second: 4.775, epoch: 12.5[0m
[32m[2022-08-31 16:13:39,476] [    INFO][0m - loss: 0.08231056, learning_rate: 2.22e-05, global_step: 260, interval_runtime: 1.9333, interval_samples_per_second: 4.138, interval_steps_per_second: 5.173, epoch: 13.0[0m
[32m[2022-08-31 16:13:41,535] [    INFO][0m - loss: 0.11971277, learning_rate: 2.19e-05, global_step: 270, interval_runtime: 2.0592, interval_samples_per_second: 3.885, interval_steps_per_second: 4.856, epoch: 13.5[0m
[32m[2022-08-31 16:13:43,468] [    INFO][0m - loss: 0.02234414, learning_rate: 2.16e-05, global_step: 280, interval_runtime: 1.9325, interval_samples_per_second: 4.14, interval_steps_per_second: 5.175, epoch: 14.0[0m
[32m[2022-08-31 16:13:45,541] [    INFO][0m - loss: 0.00470506, learning_rate: 2.13e-05, global_step: 290, interval_runtime: 2.0736, interval_samples_per_second: 3.858, interval_steps_per_second: 4.822, epoch: 14.5[0m
[32m[2022-08-31 16:13:47,490] [    INFO][0m - loss: 0.00062028, learning_rate: 2.1e-05, global_step: 300, interval_runtime: 1.9482, interval_samples_per_second: 4.106, interval_steps_per_second: 5.133, epoch: 15.0[0m
[32m[2022-08-31 16:13:47,490] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 16:13:47,490] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 16:13:47,491] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 16:13:47,491] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 16:13:47,491] [    INFO][0m -   Total prediction steps = 5[0m
[32m[2022-08-31 16:13:49,276] [    INFO][0m - eval_loss: 4.089957237243652, eval_accuracy: 0.5875, eval_runtime: 1.7847, eval_samples_per_second: 89.649, eval_steps_per_second: 2.802, epoch: 15.0[0m
[32m[2022-08-31 16:13:49,276] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-300[0m
[32m[2022-08-31 16:13:49,276] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 16:13:53,214] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-300/tokenizer_config.json[0m
[32m[2022-08-31 16:13:53,214] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-300/special_tokens_map.json[0m
[32m[2022-08-31 16:14:01,053] [    INFO][0m - loss: 1.631e-05, learning_rate: 2.07e-05, global_step: 310, interval_runtime: 13.5626, interval_samples_per_second: 0.59, interval_steps_per_second: 0.737, epoch: 15.5[0m
[32m[2022-08-31 16:14:02,992] [    INFO][0m - loss: 0.20631397, learning_rate: 2.04e-05, global_step: 320, interval_runtime: 1.9385, interval_samples_per_second: 4.127, interval_steps_per_second: 5.159, epoch: 16.0[0m
[32m[2022-08-31 16:14:05,052] [    INFO][0m - loss: 0.12767844, learning_rate: 2.01e-05, global_step: 330, interval_runtime: 2.06, interval_samples_per_second: 3.884, interval_steps_per_second: 4.854, epoch: 16.5[0m
[32m[2022-08-31 16:14:06,990] [    INFO][0m - loss: 0.00098157, learning_rate: 1.98e-05, global_step: 340, interval_runtime: 1.9394, interval_samples_per_second: 4.125, interval_steps_per_second: 5.156, epoch: 17.0[0m
[32m[2022-08-31 16:14:09,052] [    INFO][0m - loss: 9.253e-05, learning_rate: 1.95e-05, global_step: 350, interval_runtime: 2.0613, interval_samples_per_second: 3.881, interval_steps_per_second: 4.851, epoch: 17.5[0m
[32m[2022-08-31 16:14:10,998] [    INFO][0m - loss: 0.00969649, learning_rate: 1.9200000000000003e-05, global_step: 360, interval_runtime: 1.9456, interval_samples_per_second: 4.112, interval_steps_per_second: 5.14, epoch: 18.0[0m
[32m[2022-08-31 16:14:13,100] [    INFO][0m - loss: 8.021e-05, learning_rate: 1.8900000000000002e-05, global_step: 370, interval_runtime: 2.103, interval_samples_per_second: 3.804, interval_steps_per_second: 4.755, epoch: 18.5[0m
[32m[2022-08-31 16:14:15,037] [    INFO][0m - loss: 8.054e-05, learning_rate: 1.86e-05, global_step: 380, interval_runtime: 1.9364, interval_samples_per_second: 4.131, interval_steps_per_second: 5.164, epoch: 19.0[0m
[32m[2022-08-31 16:14:17,106] [    INFO][0m - loss: 0.10062968, learning_rate: 1.83e-05, global_step: 390, interval_runtime: 2.0695, interval_samples_per_second: 3.866, interval_steps_per_second: 4.832, epoch: 19.5[0m
[32m[2022-08-31 16:14:19,051] [    INFO][0m - loss: 0.00037318, learning_rate: 1.8e-05, global_step: 400, interval_runtime: 1.9446, interval_samples_per_second: 4.114, interval_steps_per_second: 5.143, epoch: 20.0[0m
[32m[2022-08-31 16:14:19,052] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 16:14:19,052] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 16:14:19,052] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 16:14:19,052] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 16:14:19,052] [    INFO][0m -   Total prediction steps = 5[0m
[32m[2022-08-31 16:14:20,880] [    INFO][0m - eval_loss: 4.835362434387207, eval_accuracy: 0.55, eval_runtime: 1.8277, eval_samples_per_second: 87.541, eval_steps_per_second: 2.736, epoch: 20.0[0m
[32m[2022-08-31 16:14:20,881] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-400[0m
[32m[2022-08-31 16:14:20,881] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 16:14:24,470] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-400/tokenizer_config.json[0m
[32m[2022-08-31 16:14:24,471] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-400/special_tokens_map.json[0m
[32m[2022-08-31 16:14:32,116] [    INFO][0m - loss: 0.00104261, learning_rate: 1.77e-05, global_step: 410, interval_runtime: 13.0649, interval_samples_per_second: 0.612, interval_steps_per_second: 0.765, epoch: 20.5[0m
[32m[2022-08-31 16:14:34,068] [    INFO][0m - loss: 6.096e-05, learning_rate: 1.74e-05, global_step: 420, interval_runtime: 1.9522, interval_samples_per_second: 4.098, interval_steps_per_second: 5.123, epoch: 21.0[0m
[32m[2022-08-31 16:14:36,132] [    INFO][0m - loss: 4.404e-05, learning_rate: 1.71e-05, global_step: 430, interval_runtime: 2.0644, interval_samples_per_second: 3.875, interval_steps_per_second: 4.844, epoch: 21.5[0m
[32m[2022-08-31 16:14:38,073] [    INFO][0m - loss: 3.039e-05, learning_rate: 1.6800000000000002e-05, global_step: 440, interval_runtime: 1.9406, interval_samples_per_second: 4.122, interval_steps_per_second: 5.153, epoch: 22.0[0m
[32m[2022-08-31 16:14:40,136] [    INFO][0m - loss: 4.432e-05, learning_rate: 1.65e-05, global_step: 450, interval_runtime: 2.0628, interval_samples_per_second: 3.878, interval_steps_per_second: 4.848, epoch: 22.5[0m
[32m[2022-08-31 16:14:42,081] [    INFO][0m - loss: 2.164e-05, learning_rate: 1.62e-05, global_step: 460, interval_runtime: 1.946, interval_samples_per_second: 4.111, interval_steps_per_second: 5.139, epoch: 23.0[0m
[32m[2022-08-31 16:14:44,172] [    INFO][0m - loss: 2.174e-05, learning_rate: 1.59e-05, global_step: 470, interval_runtime: 2.0905, interval_samples_per_second: 3.827, interval_steps_per_second: 4.784, epoch: 23.5[0m
[32m[2022-08-31 16:14:46,111] [    INFO][0m - loss: 6.789e-05, learning_rate: 1.56e-05, global_step: 480, interval_runtime: 1.939, interval_samples_per_second: 4.126, interval_steps_per_second: 5.157, epoch: 24.0[0m
[32m[2022-08-31 16:14:48,202] [    INFO][0m - loss: 1.9e-05, learning_rate: 1.53e-05, global_step: 490, interval_runtime: 2.0913, interval_samples_per_second: 3.825, interval_steps_per_second: 4.782, epoch: 24.5[0m
[32m[2022-08-31 16:14:50,153] [    INFO][0m - loss: 2.033e-05, learning_rate: 1.5e-05, global_step: 500, interval_runtime: 1.9509, interval_samples_per_second: 4.101, interval_steps_per_second: 5.126, epoch: 25.0[0m
[32m[2022-08-31 16:14:50,154] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 16:14:50,154] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 16:14:50,154] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 16:14:50,154] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 16:14:50,154] [    INFO][0m -   Total prediction steps = 5[0m
[32m[2022-08-31 16:14:52,008] [    INFO][0m - eval_loss: 5.129179000854492, eval_accuracy: 0.56875, eval_runtime: 1.854, eval_samples_per_second: 86.298, eval_steps_per_second: 2.697, epoch: 25.0[0m
[32m[2022-08-31 16:14:52,009] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-500[0m
[32m[2022-08-31 16:14:52,009] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 16:14:55,530] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-500/tokenizer_config.json[0m
[32m[2022-08-31 16:14:55,531] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-500/special_tokens_map.json[0m
[32m[2022-08-31 16:15:03,236] [    INFO][0m - loss: 1.959e-05, learning_rate: 1.47e-05, global_step: 510, interval_runtime: 13.0827, interval_samples_per_second: 0.611, interval_steps_per_second: 0.764, epoch: 25.5[0m
[32m[2022-08-31 16:15:05,168] [    INFO][0m - loss: 5.367e-05, learning_rate: 1.44e-05, global_step: 520, interval_runtime: 1.9319, interval_samples_per_second: 4.141, interval_steps_per_second: 5.176, epoch: 26.0[0m
[32m[2022-08-31 16:15:07,249] [    INFO][0m - loss: 4.56e-06, learning_rate: 1.4099999999999999e-05, global_step: 530, interval_runtime: 2.0811, interval_samples_per_second: 3.844, interval_steps_per_second: 4.805, epoch: 26.5[0m
[32m[2022-08-31 16:15:09,187] [    INFO][0m - loss: 1.906e-05, learning_rate: 1.3800000000000002e-05, global_step: 540, interval_runtime: 1.9381, interval_samples_per_second: 4.128, interval_steps_per_second: 5.16, epoch: 27.0[0m
[32m[2022-08-31 16:15:11,308] [    INFO][0m - loss: 5.138e-05, learning_rate: 1.3500000000000001e-05, global_step: 550, interval_runtime: 2.1205, interval_samples_per_second: 3.773, interval_steps_per_second: 4.716, epoch: 27.5[0m
[32m[2022-08-31 16:15:13,282] [    INFO][0m - loss: 8.1e-06, learning_rate: 1.32e-05, global_step: 560, interval_runtime: 1.9745, interval_samples_per_second: 4.052, interval_steps_per_second: 5.065, epoch: 28.0[0m
[32m[2022-08-31 16:15:15,445] [    INFO][0m - loss: 4.9e-06, learning_rate: 1.29e-05, global_step: 570, interval_runtime: 2.0735, interval_samples_per_second: 3.858, interval_steps_per_second: 4.823, epoch: 28.5[0m
[32m[2022-08-31 16:15:17,400] [    INFO][0m - loss: 7.58e-06, learning_rate: 1.26e-05, global_step: 580, interval_runtime: 2.045, interval_samples_per_second: 3.912, interval_steps_per_second: 4.89, epoch: 29.0[0m
[32m[2022-08-31 16:15:19,491] [    INFO][0m - loss: 4.01e-06, learning_rate: 1.2299999999999999e-05, global_step: 590, interval_runtime: 2.0899, interval_samples_per_second: 3.828, interval_steps_per_second: 4.785, epoch: 29.5[0m
[32m[2022-08-31 16:15:21,446] [    INFO][0m - loss: 5.81e-06, learning_rate: 1.2e-05, global_step: 600, interval_runtime: 1.955, interval_samples_per_second: 4.092, interval_steps_per_second: 5.115, epoch: 30.0[0m
[32m[2022-08-31 16:15:21,447] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 16:15:21,447] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 16:15:21,447] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 16:15:21,447] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 16:15:21,448] [    INFO][0m -   Total prediction steps = 5[0m
[32m[2022-08-31 16:15:23,280] [    INFO][0m - eval_loss: 5.451386451721191, eval_accuracy: 0.5625, eval_runtime: 1.8318, eval_samples_per_second: 87.344, eval_steps_per_second: 2.729, epoch: 30.0[0m
[32m[2022-08-31 16:15:23,280] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-600[0m
[32m[2022-08-31 16:15:23,281] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 16:15:26,561] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-600/tokenizer_config.json[0m
[32m[2022-08-31 16:15:26,562] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-600/special_tokens_map.json[0m
[32m[2022-08-31 16:15:34,285] [    INFO][0m - loss: 5.25e-06, learning_rate: 1.1700000000000001e-05, global_step: 610, interval_runtime: 12.8398, interval_samples_per_second: 0.623, interval_steps_per_second: 0.779, epoch: 30.5[0m
[32m[2022-08-31 16:15:36,222] [    INFO][0m - loss: 5.23e-06, learning_rate: 1.1400000000000001e-05, global_step: 620, interval_runtime: 1.9369, interval_samples_per_second: 4.13, interval_steps_per_second: 5.163, epoch: 31.0[0m
[32m[2022-08-31 16:15:38,304] [    INFO][0m - loss: 0.00176634, learning_rate: 1.11e-05, global_step: 630, interval_runtime: 2.0812, interval_samples_per_second: 3.844, interval_steps_per_second: 4.805, epoch: 31.5[0m
[32m[2022-08-31 16:15:40,239] [    INFO][0m - loss: 3.36e-06, learning_rate: 1.08e-05, global_step: 640, interval_runtime: 1.9352, interval_samples_per_second: 4.134, interval_steps_per_second: 5.168, epoch: 32.0[0m
[32m[2022-08-31 16:15:42,307] [    INFO][0m - loss: 9.67e-06, learning_rate: 1.05e-05, global_step: 650, interval_runtime: 2.0682, interval_samples_per_second: 3.868, interval_steps_per_second: 4.835, epoch: 32.5[0m
[32m[2022-08-31 16:15:44,258] [    INFO][0m - loss: 0.00019306, learning_rate: 1.02e-05, global_step: 660, interval_runtime: 1.9509, interval_samples_per_second: 4.101, interval_steps_per_second: 5.126, epoch: 33.0[0m
[32m[2022-08-31 16:15:46,337] [    INFO][0m - loss: 4.23e-06, learning_rate: 9.9e-06, global_step: 670, interval_runtime: 2.0792, interval_samples_per_second: 3.848, interval_steps_per_second: 4.81, epoch: 33.5[0m
[32m[2022-08-31 16:15:48,292] [    INFO][0m - loss: 5.21e-06, learning_rate: 9.600000000000001e-06, global_step: 680, interval_runtime: 1.9547, interval_samples_per_second: 4.093, interval_steps_per_second: 5.116, epoch: 34.0[0m
[32m[2022-08-31 16:15:50,373] [    INFO][0m - loss: 3.15e-06, learning_rate: 9.3e-06, global_step: 690, interval_runtime: 2.0816, interval_samples_per_second: 3.843, interval_steps_per_second: 4.804, epoch: 34.5[0m
[32m[2022-08-31 16:15:52,315] [    INFO][0m - loss: 2.57e-06, learning_rate: 9e-06, global_step: 700, interval_runtime: 1.9411, interval_samples_per_second: 4.121, interval_steps_per_second: 5.152, epoch: 35.0[0m
[32m[2022-08-31 16:15:52,315] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 16:15:52,316] [    INFO][0m -   Num examples = 160[0m
[32m[2022-08-31 16:15:52,316] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 16:15:52,316] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 16:15:52,316] [    INFO][0m -   Total prediction steps = 5[0m
[32m[2022-08-31 16:15:54,130] [    INFO][0m - eval_loss: 5.665124416351318, eval_accuracy: 0.53125, eval_runtime: 1.8136, eval_samples_per_second: 88.222, eval_steps_per_second: 2.757, epoch: 35.0[0m
[32m[2022-08-31 16:15:54,130] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-700[0m
[32m[2022-08-31 16:15:54,131] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 16:15:57,341] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-700/tokenizer_config.json[0m
[32m[2022-08-31 16:15:57,342] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-700/special_tokens_map.json[0m
[32m[2022-08-31 16:16:02,858] [    INFO][0m - 
Training completed. 
[0m
[32m[2022-08-31 16:16:02,858] [    INFO][0m - Loading best model from ./checkpoints/checkpoint-300 (score: 0.5875).[0m
[32m[2022-08-31 16:16:03,734] [    INFO][0m - train_runtime: 218.7685, train_samples_per_second: 36.568, train_steps_per_second: 4.571, train_loss: 0.11297184806328785, epoch: 35.0[0m
[32m[2022-08-31 16:16:03,758] [    INFO][0m - Saving model checkpoint to ./checkpoints/[0m
[32m[2022-08-31 16:16:03,759] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 16:16:06,977] [    INFO][0m - tokenizer config file saved in ./checkpoints/tokenizer_config.json[0m
[32m[2022-08-31 16:16:06,978] [    INFO][0m - Special tokens file saved in ./checkpoints/special_tokens_map.json[0m
[32m[2022-08-31 16:16:06,979] [    INFO][0m - ***** train metrics *****[0m
[32m[2022-08-31 16:16:06,979] [    INFO][0m -   epoch                    =       35.0[0m
[32m[2022-08-31 16:16:06,979] [    INFO][0m -   train_loss               =      0.113[0m
[32m[2022-08-31 16:16:06,979] [    INFO][0m -   train_runtime            = 0:03:38.76[0m
[32m[2022-08-31 16:16:06,979] [    INFO][0m -   train_samples_per_second =     36.568[0m
[32m[2022-08-31 16:16:06,980] [    INFO][0m -   train_steps_per_second   =      4.571[0m
[32m[2022-08-31 16:16:06,983] [    INFO][0m - ***** Running Prediction *****[0m
[32m[2022-08-31 16:16:06,984] [    INFO][0m -   Num examples = 2838[0m
[32m[2022-08-31 16:16:06,984] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 16:16:06,984] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 16:16:06,984] [    INFO][0m -   Total prediction steps = 89[0m
[32m[2022-08-31 16:16:39,425] [    INFO][0m - ***** test metrics *****[0m
[32m[2022-08-31 16:16:39,425] [    INFO][0m -   test_accuracy           =     0.5208[0m
[32m[2022-08-31 16:16:39,426] [    INFO][0m -   test_loss               =     4.7111[0m
[32m[2022-08-31 16:16:39,426] [    INFO][0m -   test_runtime            = 0:00:32.44[0m
[32m[2022-08-31 16:16:39,426] [    INFO][0m -   test_samples_per_second =     87.481[0m
[32m[2022-08-31 16:16:39,426] [    INFO][0m -   test_steps_per_second   =      2.743[0m
[32m[2022-08-31 16:16:39,426] [    INFO][0m - ***** Running Prediction *****[0m
[32m[2022-08-31 16:16:39,427] [    INFO][0m -   Num examples = 3000[0m
[32m[2022-08-31 16:16:39,427] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 16:16:39,427] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 16:16:39,427] [    INFO][0m -   Total prediction steps = 94[0m
[32m[2022-08-31 16:17:22,395] [    INFO][0m - Predictions for cslf saved to ./fewclue_submit_examples.[0m
[]
run.sh: line 73: --freeze_plm: command not found
