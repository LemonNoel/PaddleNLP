[33m[2022-08-31 14:25:37,271] [ WARNING][0m - evaluation_strategy reset to IntervalStrategy.STEPS for do_eval is True. you can also set evaluation_strategy='epoch'.[0m
[32m[2022-08-31 14:25:37,272] [    INFO][0m - The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).[0m
[32m[2022-08-31 14:25:37,272] [    INFO][0m - ============================================================[0m
[32m[2022-08-31 14:25:37,272] [    INFO][0m -      Model Configuration Arguments      [0m
[32m[2022-08-31 14:25:37,272] [    INFO][0m - paddle commit id              :65f388690048d0965ec7d3b43fb6ee9d8c6dee7c[0m
[32m[2022-08-31 14:25:37,273] [    INFO][0m - do_save                       :True[0m
[32m[2022-08-31 14:25:37,273] [    INFO][0m - do_test                       :True[0m
[32m[2022-08-31 14:25:37,273] [    INFO][0m - early_stop_patience           :4[0m
[32m[2022-08-31 14:25:37,273] [    INFO][0m - export_type                   :paddle[0m
[32m[2022-08-31 14:25:37,273] [    INFO][0m - model_name_or_path            :ernie-3.0-base-zh[0m
[32m[2022-08-31 14:25:37,273] [    INFO][0m - [0m
[32m[2022-08-31 14:25:37,273] [    INFO][0m - ============================================================[0m
[32m[2022-08-31 14:25:37,274] [    INFO][0m -       Data Configuration Arguments      [0m
[32m[2022-08-31 14:25:37,274] [    INFO][0m - paddle commit id              :65f388690048d0965ec7d3b43fb6ee9d8c6dee7c[0m
[32m[2022-08-31 14:25:37,274] [    INFO][0m - encoder_hidden_size           :200[0m
[32m[2022-08-31 14:25:37,274] [    INFO][0m - prompt                        :“{'text':'text_a'}”和“{'text':'text_b'}”之间的逻辑关系是{'mask'}{'mask'}。[0m
[32m[2022-08-31 14:25:37,274] [    INFO][0m - soft_encoder                  :lstm[0m
[32m[2022-08-31 14:25:37,274] [    INFO][0m - split_id                      :few_all[0m
[32m[2022-08-31 14:25:37,274] [    INFO][0m - task_name                     :cmnli[0m
[32m[2022-08-31 14:25:37,275] [    INFO][0m - [0m
[32m[2022-08-31 14:25:37,275] [    INFO][0m - Already cached /ssd2/wanghuijuan03/.paddlenlp/models/ernie-3.0-base-zh/ernie_3.0_base_zh.pdparams[0m
W0831 14:25:37.277127 32797 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 11.2
W0831 14:25:37.283023 32797 gpu_resources.cc:91] device: 0, cuDNN Version: 8.1.
[32m[2022-08-31 14:25:40,361] [    INFO][0m - Already cached /ssd2/wanghuijuan03/.paddlenlp/models/ernie-3.0-base-zh/ernie_3.0_base_zh_vocab.txt[0m
[32m[2022-08-31 14:25:40,392] [    INFO][0m - tokenizer config file saved in /ssd2/wanghuijuan03/.paddlenlp/models/ernie-3.0-base-zh/tokenizer_config.json[0m
[32m[2022-08-31 14:25:40,393] [    INFO][0m - Special tokens file saved in /ssd2/wanghuijuan03/.paddlenlp/models/ernie-3.0-base-zh/special_tokens_map.json[0m
[32m[2022-08-31 14:25:40,394] [    INFO][0m - Using template: [{'add_prefix_space': '', 'hard': '“'}, {'add_prefix_space': '', 'text': 'text_a'}, {'add_prefix_space': '', 'hard': '”和“'}, {'add_prefix_space': '', 'text': 'text_b'}, {'add_prefix_space': '', 'hard': '”之间的逻辑关系是'}, {'add_prefix_space': '', 'mask': None}, {'add_prefix_space': '', 'mask': None}, {'add_prefix_space': '', 'hard': '。'}][0m
[32m[2022-08-31 14:25:40,397] [    INFO][0m - {'contradiction': 0, 'entailment': 1, 'neutral': 2}[0m
2022-08-31 14:25:40,399 INFO [download.py:119] unique_endpoints {''}
2022-08-31 14:25:42,696 INFO [download.py:119] unique_endpoints {''}
[32m[2022-08-31 14:25:42,858] [    INFO][0m - ============================================================[0m
[32m[2022-08-31 14:25:42,858] [    INFO][0m -     Training Configuration Arguments    [0m
[32m[2022-08-31 14:25:42,858] [    INFO][0m - paddle commit id              :65f388690048d0965ec7d3b43fb6ee9d8c6dee7c[0m
[32m[2022-08-31 14:25:42,858] [    INFO][0m - _no_sync_in_gradient_accumulation:True[0m
[32m[2022-08-31 14:25:42,858] [    INFO][0m - adam_beta1                    :0.9[0m
[32m[2022-08-31 14:25:42,858] [    INFO][0m - adam_beta2                    :0.999[0m
[32m[2022-08-31 14:25:42,858] [    INFO][0m - adam_epsilon                  :1e-08[0m
[32m[2022-08-31 14:25:42,859] [    INFO][0m - alpha_rdrop                   :5.0[0m
[32m[2022-08-31 14:25:42,859] [    INFO][0m - alpha_rgl                     :0.5[0m
[32m[2022-08-31 14:25:42,859] [    INFO][0m - current_device                :gpu:0[0m
[32m[2022-08-31 14:25:42,859] [    INFO][0m - dataloader_drop_last          :False[0m
[32m[2022-08-31 14:25:42,859] [    INFO][0m - dataloader_num_workers        :0[0m
[32m[2022-08-31 14:25:42,859] [    INFO][0m - device                        :gpu[0m
[32m[2022-08-31 14:25:42,859] [    INFO][0m - disable_tqdm                  :True[0m
[32m[2022-08-31 14:25:42,859] [    INFO][0m - do_eval                       :True[0m
[32m[2022-08-31 14:25:42,859] [    INFO][0m - do_export                     :False[0m
[32m[2022-08-31 14:25:42,859] [    INFO][0m - do_predict                    :True[0m
[32m[2022-08-31 14:25:42,859] [    INFO][0m - do_train                      :True[0m
[32m[2022-08-31 14:25:42,859] [    INFO][0m - eval_batch_size               :64[0m
[32m[2022-08-31 14:25:42,859] [    INFO][0m - eval_steps                    :100[0m
[32m[2022-08-31 14:25:42,860] [    INFO][0m - evaluation_strategy           :IntervalStrategy.STEPS[0m
[32m[2022-08-31 14:25:42,860] [    INFO][0m - first_max_length              :None[0m
[32m[2022-08-31 14:25:42,860] [    INFO][0m - fp16                          :False[0m
[32m[2022-08-31 14:25:42,860] [    INFO][0m - fp16_opt_level                :O1[0m
[32m[2022-08-31 14:25:42,860] [    INFO][0m - freeze_dropout                :False[0m
[32m[2022-08-31 14:25:42,860] [    INFO][0m - freeze_plm                    :False[0m
[32m[2022-08-31 14:25:42,860] [    INFO][0m - gradient_accumulation_steps   :1[0m
[32m[2022-08-31 14:25:42,860] [    INFO][0m - greater_is_better             :True[0m
[32m[2022-08-31 14:25:42,860] [    INFO][0m - ignore_data_skip              :False[0m
[32m[2022-08-31 14:25:42,860] [    INFO][0m - label_names                   :None[0m
[32m[2022-08-31 14:25:42,860] [    INFO][0m - learning_rate                 :3e-05[0m
[32m[2022-08-31 14:25:42,860] [    INFO][0m - load_best_model_at_end        :True[0m
[32m[2022-08-31 14:25:42,860] [    INFO][0m - local_process_index           :0[0m
[32m[2022-08-31 14:25:42,861] [    INFO][0m - local_rank                    :-1[0m
[32m[2022-08-31 14:25:42,861] [    INFO][0m - log_level                     :-1[0m
[32m[2022-08-31 14:25:42,861] [    INFO][0m - log_level_replica             :-1[0m
[32m[2022-08-31 14:25:42,861] [    INFO][0m - log_on_each_node              :True[0m
[32m[2022-08-31 14:25:42,861] [    INFO][0m - logging_dir                   :./checkpoints/runs/Aug31_14-25-37_instance-3bwob41y-01[0m
[32m[2022-08-31 14:25:42,861] [    INFO][0m - logging_first_step            :False[0m
[32m[2022-08-31 14:25:42,861] [    INFO][0m - logging_steps                 :10[0m
[32m[2022-08-31 14:25:42,861] [    INFO][0m - logging_strategy              :IntervalStrategy.STEPS[0m
[32m[2022-08-31 14:25:42,861] [    INFO][0m - lr_scheduler_type             :SchedulerType.LINEAR[0m
[32m[2022-08-31 14:25:42,861] [    INFO][0m - max_grad_norm                 :1.0[0m
[32m[2022-08-31 14:25:42,861] [    INFO][0m - max_seq_length                :128[0m
[32m[2022-08-31 14:25:42,861] [    INFO][0m - max_steps                     :-1[0m
[32m[2022-08-31 14:25:42,861] [    INFO][0m - metric_for_best_model         :accuracy[0m
[32m[2022-08-31 14:25:42,862] [    INFO][0m - minimum_eval_times            :None[0m
[32m[2022-08-31 14:25:42,862] [    INFO][0m - no_cuda                       :False[0m
[32m[2022-08-31 14:25:42,862] [    INFO][0m - num_train_epochs              :50.0[0m
[32m[2022-08-31 14:25:42,862] [    INFO][0m - optim                         :OptimizerNames.ADAMW[0m
[32m[2022-08-31 14:25:42,862] [    INFO][0m - other_max_length              :None[0m
[32m[2022-08-31 14:25:42,862] [    INFO][0m - output_dir                    :./checkpoints/[0m
[32m[2022-08-31 14:25:42,862] [    INFO][0m - overwrite_output_dir          :False[0m
[32m[2022-08-31 14:25:42,862] [    INFO][0m - past_index                    :-1[0m
[32m[2022-08-31 14:25:42,862] [    INFO][0m - per_device_eval_batch_size    :64[0m
[32m[2022-08-31 14:25:42,862] [    INFO][0m - per_device_train_batch_size   :64[0m
[32m[2022-08-31 14:25:42,862] [    INFO][0m - ppt_adam_beta1                :0.9[0m
[32m[2022-08-31 14:25:42,862] [    INFO][0m - ppt_adam_beta2                :0.999[0m
[32m[2022-08-31 14:25:42,862] [    INFO][0m - ppt_adam_epsilon              :1e-08[0m
[32m[2022-08-31 14:25:42,862] [    INFO][0m - ppt_learning_rate             :0.0003[0m
[32m[2022-08-31 14:25:42,863] [    INFO][0m - ppt_weight_decay              :0.0[0m
[32m[2022-08-31 14:25:42,863] [    INFO][0m - prediction_loss_only          :False[0m
[32m[2022-08-31 14:25:42,863] [    INFO][0m - process_index                 :0[0m
[32m[2022-08-31 14:25:42,863] [    INFO][0m - remove_unused_columns         :True[0m
[32m[2022-08-31 14:25:42,863] [    INFO][0m - report_to                     :['visualdl'][0m
[32m[2022-08-31 14:25:42,863] [    INFO][0m - resume_from_checkpoint        :None[0m
[32m[2022-08-31 14:25:42,863] [    INFO][0m - run_name                      :./checkpoints/[0m
[32m[2022-08-31 14:25:42,863] [    INFO][0m - save_on_each_node             :False[0m
[32m[2022-08-31 14:25:42,863] [    INFO][0m - save_steps                    :100[0m
[32m[2022-08-31 14:25:42,863] [    INFO][0m - save_strategy                 :IntervalStrategy.STEPS[0m
[32m[2022-08-31 14:25:42,863] [    INFO][0m - save_total_limit              :None[0m
[32m[2022-08-31 14:25:42,863] [    INFO][0m - scale_loss                    :32768[0m
[32m[2022-08-31 14:25:42,863] [    INFO][0m - seed                          :42[0m
[32m[2022-08-31 14:25:42,863] [    INFO][0m - should_log                    :True[0m
[32m[2022-08-31 14:25:42,864] [    INFO][0m - should_save                   :True[0m
[32m[2022-08-31 14:25:42,864] [    INFO][0m - task_type                     :multi-class[0m
[32m[2022-08-31 14:25:42,864] [    INFO][0m - train_batch_size              :64[0m
[32m[2022-08-31 14:25:42,864] [    INFO][0m - truncate_mode                 :tail[0m
[32m[2022-08-31 14:25:42,864] [    INFO][0m - use_rdrop                     :False[0m
[32m[2022-08-31 14:25:42,864] [    INFO][0m - use_rgl                       :False[0m
[32m[2022-08-31 14:25:42,864] [    INFO][0m - warmup_ratio                  :0.0[0m
[32m[2022-08-31 14:25:42,864] [    INFO][0m - warmup_steps                  :0[0m
[32m[2022-08-31 14:25:42,864] [    INFO][0m - weight_decay                  :0.0[0m
[32m[2022-08-31 14:25:42,864] [    INFO][0m - world_size                    :1[0m
[32m[2022-08-31 14:25:42,864] [    INFO][0m - [0m
[32m[2022-08-31 14:25:42,867] [    INFO][0m - ***** Running training *****[0m
[32m[2022-08-31 14:25:42,867] [    INFO][0m -   Num examples = 391783[0m
[32m[2022-08-31 14:25:42,867] [    INFO][0m -   Num Epochs = 50[0m
[32m[2022-08-31 14:25:42,867] [    INFO][0m -   Instantaneous batch size per device = 64[0m
[32m[2022-08-31 14:25:42,867] [    INFO][0m -   Total train batch size (w. parallel, distributed & accumulation) = 64[0m
[32m[2022-08-31 14:25:42,867] [    INFO][0m -   Gradient Accumulation steps = 1[0m
[32m[2022-08-31 14:25:42,867] [    INFO][0m -   Total optimization steps = 306100.0[0m
[32m[2022-08-31 14:25:42,867] [    INFO][0m -   Total num train samples = 19589150[0m
[32m[2022-08-31 14:25:49,741] [    INFO][0m - loss: 1.21554871, learning_rate: 2.9999019928128062e-05, global_step: 10, interval_runtime: 6.8718, interval_samples_per_second: 9.313, interval_steps_per_second: 1.455, epoch: 0.0016[0m
[32m[2022-08-31 14:25:55,321] [    INFO][0m - loss: 1.12669315, learning_rate: 2.9998039856256126e-05, global_step: 20, interval_runtime: 5.5807, interval_samples_per_second: 11.468, interval_steps_per_second: 1.792, epoch: 0.0033[0m
[32m[2022-08-31 14:26:00,909] [    INFO][0m - loss: 1.12475176, learning_rate: 2.9997059784384187e-05, global_step: 30, interval_runtime: 5.5884, interval_samples_per_second: 11.452, interval_steps_per_second: 1.789, epoch: 0.0049[0m
[32m[2022-08-31 14:26:06,566] [    INFO][0m - loss: 1.0891777, learning_rate: 2.999607971251225e-05, global_step: 40, interval_runtime: 5.6553, interval_samples_per_second: 11.317, interval_steps_per_second: 1.768, epoch: 0.0065[0m
[32m[2022-08-31 14:26:12,201] [    INFO][0m - loss: 1.0767786, learning_rate: 2.9995099640640313e-05, global_step: 50, interval_runtime: 5.636, interval_samples_per_second: 11.356, interval_steps_per_second: 1.774, epoch: 0.0082[0m
[32m[2022-08-31 14:26:17,785] [    INFO][0m - loss: 1.04838524, learning_rate: 2.9994119568768377e-05, global_step: 60, interval_runtime: 5.5832, interval_samples_per_second: 11.463, interval_steps_per_second: 1.791, epoch: 0.0098[0m
[32m[2022-08-31 14:26:23,363] [    INFO][0m - loss: 1.01537561, learning_rate: 2.9993139496896438e-05, global_step: 70, interval_runtime: 5.5786, interval_samples_per_second: 11.472, interval_steps_per_second: 1.793, epoch: 0.0114[0m
[32m[2022-08-31 14:26:28,964] [    INFO][0m - loss: 1.01055336, learning_rate: 2.9992159425024502e-05, global_step: 80, interval_runtime: 5.6012, interval_samples_per_second: 11.426, interval_steps_per_second: 1.785, epoch: 0.0131[0m
[32m[2022-08-31 14:26:34,549] [    INFO][0m - loss: 0.95157442, learning_rate: 2.9991179353152563e-05, global_step: 90, interval_runtime: 5.5854, interval_samples_per_second: 11.458, interval_steps_per_second: 1.79, epoch: 0.0147[0m
[32m[2022-08-31 14:26:40,140] [    INFO][0m - loss: 0.92614336, learning_rate: 2.9990199281280628e-05, global_step: 100, interval_runtime: 5.5908, interval_samples_per_second: 11.447, interval_steps_per_second: 1.789, epoch: 0.0163[0m
[32m[2022-08-31 14:26:40,140] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 14:26:40,141] [    INFO][0m -   Num examples = 12241[0m
[32m[2022-08-31 14:26:40,141] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 14:26:40,141] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 14:26:40,141] [    INFO][0m -   Total prediction steps = 192[0m
[32m[2022-08-31 14:27:35,100] [    INFO][0m - eval_loss: 0.8164545893669128, eval_accuracy: 0.6349971407564742, eval_runtime: 54.9586, eval_samples_per_second: 222.731, eval_steps_per_second: 3.494, epoch: 0.0163[0m
[32m[2022-08-31 14:27:35,100] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-100[0m
[32m[2022-08-31 14:27:35,101] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 14:27:36,757] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-100/tokenizer_config.json[0m
[32m[2022-08-31 14:27:36,757] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-100/special_tokens_map.json[0m
[32m[2022-08-31 14:27:44,887] [    INFO][0m - loss: 0.8577157, learning_rate: 2.998921920940869e-05, global_step: 110, interval_runtime: 64.7471, interval_samples_per_second: 0.988, interval_steps_per_second: 0.154, epoch: 0.018[0m
[32m[2022-08-31 14:27:50,612] [    INFO][0m - loss: 0.89079742, learning_rate: 2.9988239137536753e-05, global_step: 120, interval_runtime: 5.725, interval_samples_per_second: 11.179, interval_steps_per_second: 1.747, epoch: 0.0196[0m
[32m[2022-08-31 14:27:56,254] [    INFO][0m - loss: 0.80290632, learning_rate: 2.9987259065664818e-05, global_step: 130, interval_runtime: 5.642, interval_samples_per_second: 11.343, interval_steps_per_second: 1.772, epoch: 0.0212[0m
[32m[2022-08-31 14:28:01,873] [    INFO][0m - loss: 0.85783281, learning_rate: 2.9986278993792882e-05, global_step: 140, interval_runtime: 5.6195, interval_samples_per_second: 11.389, interval_steps_per_second: 1.78, epoch: 0.0229[0m
[32m[2022-08-31 14:28:07,502] [    INFO][0m - loss: 0.81869507, learning_rate: 2.9985298921920943e-05, global_step: 150, interval_runtime: 5.629, interval_samples_per_second: 11.37, interval_steps_per_second: 1.777, epoch: 0.0245[0m
[32m[2022-08-31 14:28:13,133] [    INFO][0m - loss: 0.80487223, learning_rate: 2.9984318850049004e-05, global_step: 160, interval_runtime: 5.63, interval_samples_per_second: 11.368, interval_steps_per_second: 1.776, epoch: 0.0261[0m
[32m[2022-08-31 14:28:18,768] [    INFO][0m - loss: 0.78805165, learning_rate: 2.998333877817707e-05, global_step: 170, interval_runtime: 5.6352, interval_samples_per_second: 11.357, interval_steps_per_second: 1.775, epoch: 0.0278[0m
[32m[2022-08-31 14:28:24,415] [    INFO][0m - loss: 0.81532526, learning_rate: 2.998235870630513e-05, global_step: 180, interval_runtime: 5.648, interval_samples_per_second: 11.331, interval_steps_per_second: 1.771, epoch: 0.0294[0m
[32m[2022-08-31 14:28:30,062] [    INFO][0m - loss: 0.75256863, learning_rate: 2.9981378634433194e-05, global_step: 190, interval_runtime: 5.6464, interval_samples_per_second: 11.335, interval_steps_per_second: 1.771, epoch: 0.031[0m
[32m[2022-08-31 14:28:35,733] [    INFO][0m - loss: 0.77374105, learning_rate: 2.9980398562561255e-05, global_step: 200, interval_runtime: 5.6709, interval_samples_per_second: 11.286, interval_steps_per_second: 1.763, epoch: 0.0327[0m
[32m[2022-08-31 14:28:35,733] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 14:28:35,733] [    INFO][0m -   Num examples = 12241[0m
[32m[2022-08-31 14:28:35,734] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 14:28:35,734] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 14:28:35,734] [    INFO][0m -   Total prediction steps = 192[0m
[32m[2022-08-31 14:29:30,843] [    INFO][0m - eval_loss: 0.6790741682052612, eval_accuracy: 0.7116248672494078, eval_runtime: 55.1084, eval_samples_per_second: 222.126, eval_steps_per_second: 3.484, epoch: 0.0327[0m
[32m[2022-08-31 14:29:30,843] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-200[0m
[32m[2022-08-31 14:29:30,844] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 14:29:32,522] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-200/tokenizer_config.json[0m
[32m[2022-08-31 14:29:32,523] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-200/special_tokens_map.json[0m
[32m[2022-08-31 14:29:40,733] [    INFO][0m - loss: 0.73817816, learning_rate: 2.997941849068932e-05, global_step: 210, interval_runtime: 64.9997, interval_samples_per_second: 0.985, interval_steps_per_second: 0.154, epoch: 0.0343[0m
[32m[2022-08-31 14:29:46,452] [    INFO][0m - loss: 0.7480824, learning_rate: 2.997843841881738e-05, global_step: 220, interval_runtime: 5.7191, interval_samples_per_second: 11.191, interval_steps_per_second: 1.749, epoch: 0.0359[0m
[32m[2022-08-31 14:29:52,117] [    INFO][0m - loss: 0.77800169, learning_rate: 2.9977458346945445e-05, global_step: 230, interval_runtime: 5.6652, interval_samples_per_second: 11.297, interval_steps_per_second: 1.765, epoch: 0.0376[0m
[32m[2022-08-31 14:29:57,790] [    INFO][0m - loss: 0.67297468, learning_rate: 2.9976478275073506e-05, global_step: 240, interval_runtime: 5.6726, interval_samples_per_second: 11.282, interval_steps_per_second: 1.763, epoch: 0.0392[0m
[32m[2022-08-31 14:30:03,506] [    INFO][0m - loss: 0.7737771, learning_rate: 2.997549820320157e-05, global_step: 250, interval_runtime: 5.7166, interval_samples_per_second: 11.196, interval_steps_per_second: 1.749, epoch: 0.0408[0m
[32m[2022-08-31 14:30:09,180] [    INFO][0m - loss: 0.70870543, learning_rate: 2.997451813132963e-05, global_step: 260, interval_runtime: 5.674, interval_samples_per_second: 11.28, interval_steps_per_second: 1.762, epoch: 0.0425[0m
[32m[2022-08-31 14:30:14,887] [    INFO][0m - loss: 0.75408697, learning_rate: 2.9973538059457695e-05, global_step: 270, interval_runtime: 5.7066, interval_samples_per_second: 11.215, interval_steps_per_second: 1.752, epoch: 0.0441[0m
[32m[2022-08-31 14:30:20,549] [    INFO][0m - loss: 0.73531466, learning_rate: 2.9972557987585756e-05, global_step: 280, interval_runtime: 5.6627, interval_samples_per_second: 11.302, interval_steps_per_second: 1.766, epoch: 0.0457[0m
[32m[2022-08-31 14:30:26,219] [    INFO][0m - loss: 0.71841607, learning_rate: 2.9971577915713817e-05, global_step: 290, interval_runtime: 5.6698, interval_samples_per_second: 11.288, interval_steps_per_second: 1.764, epoch: 0.0474[0m
[32m[2022-08-31 14:30:31,868] [    INFO][0m - loss: 0.67825623, learning_rate: 2.9970597843841882e-05, global_step: 300, interval_runtime: 5.6483, interval_samples_per_second: 11.331, interval_steps_per_second: 1.77, epoch: 0.049[0m
[32m[2022-08-31 14:30:31,868] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 14:30:31,869] [    INFO][0m -   Num examples = 12241[0m
[32m[2022-08-31 14:30:31,869] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 14:30:31,869] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 14:30:31,869] [    INFO][0m -   Total prediction steps = 192[0m
[32m[2022-08-31 14:31:26,687] [    INFO][0m - eval_loss: 0.6830440163612366, eval_accuracy: 0.7282901723715383, eval_runtime: 54.8178, eval_samples_per_second: 223.303, eval_steps_per_second: 3.503, epoch: 0.049[0m
[32m[2022-08-31 14:31:26,688] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-300[0m
[32m[2022-08-31 14:31:26,688] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 14:31:28,353] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-300/tokenizer_config.json[0m
[32m[2022-08-31 14:31:28,353] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-300/special_tokens_map.json[0m
[32m[2022-08-31 14:31:36,423] [    INFO][0m - loss: 0.72071791, learning_rate: 2.9969617771969943e-05, global_step: 310, interval_runtime: 64.5552, interval_samples_per_second: 0.991, interval_steps_per_second: 0.155, epoch: 0.0506[0m
[32m[2022-08-31 14:31:42,090] [    INFO][0m - loss: 0.6745223, learning_rate: 2.9968637700098007e-05, global_step: 320, interval_runtime: 5.6669, interval_samples_per_second: 11.294, interval_steps_per_second: 1.765, epoch: 0.0523[0m
[32m[2022-08-31 14:31:47,763] [    INFO][0m - loss: 0.6990756, learning_rate: 2.9967657628226068e-05, global_step: 330, interval_runtime: 5.6737, interval_samples_per_second: 11.28, interval_steps_per_second: 1.763, epoch: 0.0539[0m
[32m[2022-08-31 14:31:53,430] [    INFO][0m - loss: 0.69066486, learning_rate: 2.9966677556354133e-05, global_step: 340, interval_runtime: 5.6668, interval_samples_per_second: 11.294, interval_steps_per_second: 1.765, epoch: 0.0555[0m
[32m[2022-08-31 14:31:59,102] [    INFO][0m - loss: 0.65782485, learning_rate: 2.9965697484482194e-05, global_step: 350, interval_runtime: 5.6712, interval_samples_per_second: 11.285, interval_steps_per_second: 1.763, epoch: 0.0572[0m
[32m[2022-08-31 14:32:04,759] [    INFO][0m - loss: 0.61550908, learning_rate: 2.9964717412610258e-05, global_step: 360, interval_runtime: 5.6573, interval_samples_per_second: 11.313, interval_steps_per_second: 1.768, epoch: 0.0588[0m
[32m[2022-08-31 14:32:10,421] [    INFO][0m - loss: 0.68507719, learning_rate: 2.9963737340738322e-05, global_step: 370, interval_runtime: 5.6626, interval_samples_per_second: 11.302, interval_steps_per_second: 1.766, epoch: 0.0604[0m
[32m[2022-08-31 14:32:16,081] [    INFO][0m - loss: 0.72605276, learning_rate: 2.9962757268866387e-05, global_step: 380, interval_runtime: 5.6602, interval_samples_per_second: 11.307, interval_steps_per_second: 1.767, epoch: 0.0621[0m
[32m[2022-08-31 14:32:21,739] [    INFO][0m - loss: 0.66221895, learning_rate: 2.9961777196994448e-05, global_step: 390, interval_runtime: 5.657, interval_samples_per_second: 11.313, interval_steps_per_second: 1.768, epoch: 0.0637[0m
[32m[2022-08-31 14:32:27,398] [    INFO][0m - loss: 0.67764273, learning_rate: 2.9960797125122512e-05, global_step: 400, interval_runtime: 5.6589, interval_samples_per_second: 11.31, interval_steps_per_second: 1.767, epoch: 0.0653[0m
[32m[2022-08-31 14:32:27,398] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 14:32:27,399] [    INFO][0m -   Num examples = 12241[0m
[32m[2022-08-31 14:32:27,399] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 14:32:27,399] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 14:32:27,399] [    INFO][0m -   Total prediction steps = 192[0m
[32m[2022-08-31 14:33:22,492] [    INFO][0m - eval_loss: 0.7069441080093384, eval_accuracy: 0.7121150232824116, eval_runtime: 55.0929, eval_samples_per_second: 222.188, eval_steps_per_second: 3.485, epoch: 0.0653[0m
[32m[2022-08-31 14:33:22,493] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-400[0m
[32m[2022-08-31 14:33:22,493] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 14:33:23,941] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-400/tokenizer_config.json[0m
[32m[2022-08-31 14:33:23,941] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-400/special_tokens_map.json[0m
[32m[2022-08-31 14:33:31,865] [    INFO][0m - loss: 0.722543, learning_rate: 2.9959817053250573e-05, global_step: 410, interval_runtime: 64.4679, interval_samples_per_second: 0.993, interval_steps_per_second: 0.155, epoch: 0.067[0m
[32m[2022-08-31 14:33:37,527] [    INFO][0m - loss: 0.70435562, learning_rate: 2.9958836981378638e-05, global_step: 420, interval_runtime: 5.6614, interval_samples_per_second: 11.305, interval_steps_per_second: 1.766, epoch: 0.0686[0m
[32m[2022-08-31 14:33:43,189] [    INFO][0m - loss: 0.698592, learning_rate: 2.99578569095067e-05, global_step: 430, interval_runtime: 5.6624, interval_samples_per_second: 11.303, interval_steps_per_second: 1.766, epoch: 0.0702[0m
[32m[2022-08-31 14:33:48,876] [    INFO][0m - loss: 0.71383467, learning_rate: 2.995687683763476e-05, global_step: 440, interval_runtime: 5.6872, interval_samples_per_second: 11.253, interval_steps_per_second: 1.758, epoch: 0.0719[0m
[32m[2022-08-31 14:33:54,553] [    INFO][0m - loss: 0.64804401, learning_rate: 2.9955896765762824e-05, global_step: 450, interval_runtime: 5.6767, interval_samples_per_second: 11.274, interval_steps_per_second: 1.762, epoch: 0.0735[0m
[32m[2022-08-31 14:34:01,125] [    INFO][0m - loss: 0.66841898, learning_rate: 2.9954916693890885e-05, global_step: 460, interval_runtime: 5.6721, interval_samples_per_second: 11.283, interval_steps_per_second: 1.763, epoch: 0.0751[0m
[32m[2022-08-31 14:34:06,788] [    INFO][0m - loss: 0.69893928, learning_rate: 2.995393662201895e-05, global_step: 470, interval_runtime: 6.5634, interval_samples_per_second: 9.751, interval_steps_per_second: 1.524, epoch: 0.0768[0m
[32m[2022-08-31 14:34:12,465] [    INFO][0m - loss: 0.68817639, learning_rate: 2.995295655014701e-05, global_step: 480, interval_runtime: 5.6766, interval_samples_per_second: 11.274, interval_steps_per_second: 1.762, epoch: 0.0784[0m
[32m[2022-08-31 14:34:18,127] [    INFO][0m - loss: 0.69352393, learning_rate: 2.9951976478275075e-05, global_step: 490, interval_runtime: 5.662, interval_samples_per_second: 11.304, interval_steps_per_second: 1.766, epoch: 0.08[0m
[32m[2022-08-31 14:34:23,811] [    INFO][0m - loss: 0.62687826, learning_rate: 2.9950996406403136e-05, global_step: 500, interval_runtime: 5.6836, interval_samples_per_second: 11.26, interval_steps_per_second: 1.759, epoch: 0.0817[0m
[32m[2022-08-31 14:34:23,811] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 14:34:23,811] [    INFO][0m -   Num examples = 12241[0m
[32m[2022-08-31 14:34:23,811] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 14:34:23,811] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 14:34:23,811] [    INFO][0m -   Total prediction steps = 192[0m
[32m[2022-08-31 14:35:18,599] [    INFO][0m - eval_loss: 0.706729531288147, eval_accuracy: 0.7242872314353402, eval_runtime: 54.7868, eval_samples_per_second: 223.43, eval_steps_per_second: 3.504, epoch: 0.0817[0m
[32m[2022-08-31 14:35:18,599] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-500[0m
[32m[2022-08-31 14:35:18,599] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 14:35:20,071] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-500/tokenizer_config.json[0m
[32m[2022-08-31 14:35:20,071] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-500/special_tokens_map.json[0m
[32m[2022-08-31 14:35:28,028] [    INFO][0m - loss: 0.71502252, learning_rate: 2.99500163345312e-05, global_step: 510, interval_runtime: 64.2178, interval_samples_per_second: 0.997, interval_steps_per_second: 0.156, epoch: 0.0833[0m
[32m[2022-08-31 14:35:33,688] [    INFO][0m - loss: 0.67674956, learning_rate: 2.994903626265926e-05, global_step: 520, interval_runtime: 5.66, interval_samples_per_second: 11.307, interval_steps_per_second: 1.767, epoch: 0.0849[0m
[32m[2022-08-31 14:35:39,370] [    INFO][0m - loss: 0.75441189, learning_rate: 2.9948056190787326e-05, global_step: 530, interval_runtime: 5.6813, interval_samples_per_second: 11.265, interval_steps_per_second: 1.76, epoch: 0.0866[0m
[32m[2022-08-31 14:35:45,037] [    INFO][0m - loss: 0.66781087, learning_rate: 2.9947076118915387e-05, global_step: 540, interval_runtime: 5.6671, interval_samples_per_second: 11.293, interval_steps_per_second: 1.765, epoch: 0.0882[0m
[32m[2022-08-31 14:35:51,155] [    INFO][0m - loss: 0.63371754, learning_rate: 2.994609604704345e-05, global_step: 550, interval_runtime: 5.6614, interval_samples_per_second: 11.305, interval_steps_per_second: 1.766, epoch: 0.0898[0m
[32m[2022-08-31 14:35:56,847] [    INFO][0m - loss: 0.67632422, learning_rate: 2.9945115975171512e-05, global_step: 560, interval_runtime: 6.1485, interval_samples_per_second: 10.409, interval_steps_per_second: 1.626, epoch: 0.0915[0m
[32m[2022-08-31 14:36:02,536] [    INFO][0m - loss: 0.65212545, learning_rate: 2.9944135903299577e-05, global_step: 570, interval_runtime: 5.6893, interval_samples_per_second: 11.249, interval_steps_per_second: 1.758, epoch: 0.0931[0m
[32m[2022-08-31 14:36:08,221] [    INFO][0m - loss: 0.61424675, learning_rate: 2.9943155831427638e-05, global_step: 580, interval_runtime: 5.6854, interval_samples_per_second: 11.257, interval_steps_per_second: 1.759, epoch: 0.0947[0m
[32m[2022-08-31 14:36:13,948] [    INFO][0m - loss: 0.65015364, learning_rate: 2.99421757595557e-05, global_step: 590, interval_runtime: 5.726, interval_samples_per_second: 11.177, interval_steps_per_second: 1.746, epoch: 0.0964[0m
[32m[2022-08-31 14:36:19,606] [    INFO][0m - loss: 0.64129725, learning_rate: 2.9941195687683763e-05, global_step: 600, interval_runtime: 5.6578, interval_samples_per_second: 11.312, interval_steps_per_second: 1.767, epoch: 0.098[0m
[32m[2022-08-31 14:36:19,606] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 14:36:19,606] [    INFO][0m -   Num examples = 12241[0m
[32m[2022-08-31 14:36:19,606] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 14:36:19,606] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 14:36:19,606] [    INFO][0m -   Total prediction steps = 192[0m
[32m[2022-08-31 14:37:14,725] [    INFO][0m - eval_loss: 0.611442506313324, eval_accuracy: 0.754350134792909, eval_runtime: 55.1179, eval_samples_per_second: 222.088, eval_steps_per_second: 3.483, epoch: 0.098[0m
[32m[2022-08-31 14:37:14,725] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-600[0m
[32m[2022-08-31 14:37:14,726] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 14:37:16,198] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-600/tokenizer_config.json[0m
[32m[2022-08-31 14:37:16,198] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-600/special_tokens_map.json[0m
[32m[2022-08-31 14:37:24,718] [    INFO][0m - loss: 0.62093363, learning_rate: 2.9940215615811824e-05, global_step: 610, interval_runtime: 65.1131, interval_samples_per_second: 0.983, interval_steps_per_second: 0.154, epoch: 0.0996[0m
[32m[2022-08-31 14:37:30,375] [    INFO][0m - loss: 0.60404706, learning_rate: 2.9939235543939892e-05, global_step: 620, interval_runtime: 5.6561, interval_samples_per_second: 11.315, interval_steps_per_second: 1.768, epoch: 0.1013[0m
[32m[2022-08-31 14:37:36,039] [    INFO][0m - loss: 0.69325652, learning_rate: 2.9938255472067953e-05, global_step: 630, interval_runtime: 5.6643, interval_samples_per_second: 11.299, interval_steps_per_second: 1.765, epoch: 0.1029[0m
[32m[2022-08-31 14:37:41,743] [    INFO][0m - loss: 0.69546223, learning_rate: 2.9937275400196017e-05, global_step: 640, interval_runtime: 5.7041, interval_samples_per_second: 11.22, interval_steps_per_second: 1.753, epoch: 0.1045[0m
[32m[2022-08-31 14:37:47,411] [    INFO][0m - loss: 0.60037975, learning_rate: 2.9936295328324078e-05, global_step: 650, interval_runtime: 5.6675, interval_samples_per_second: 11.292, interval_steps_per_second: 1.764, epoch: 0.1062[0m
[32m[2022-08-31 14:37:53,099] [    INFO][0m - loss: 0.63813982, learning_rate: 2.9935315256452143e-05, global_step: 660, interval_runtime: 5.6882, interval_samples_per_second: 11.251, interval_steps_per_second: 1.758, epoch: 0.1078[0m
[32m[2022-08-31 14:37:58,760] [    INFO][0m - loss: 0.67611351, learning_rate: 2.9934335184580204e-05, global_step: 670, interval_runtime: 5.6613, interval_samples_per_second: 11.305, interval_steps_per_second: 1.766, epoch: 0.1094[0m
[32m[2022-08-31 14:38:04,412] [    INFO][0m - loss: 0.66516151, learning_rate: 2.9933355112708268e-05, global_step: 680, interval_runtime: 5.6515, interval_samples_per_second: 11.324, interval_steps_per_second: 1.769, epoch: 0.1111[0m
[32m[2022-08-31 14:38:10,084] [    INFO][0m - loss: 0.64092846, learning_rate: 2.993237504083633e-05, global_step: 690, interval_runtime: 5.6717, interval_samples_per_second: 11.284, interval_steps_per_second: 1.763, epoch: 0.1127[0m
[32m[2022-08-31 14:38:15,744] [    INFO][0m - loss: 0.61171145, learning_rate: 2.9931394968964393e-05, global_step: 700, interval_runtime: 5.6601, interval_samples_per_second: 11.307, interval_steps_per_second: 1.767, epoch: 0.1143[0m
[32m[2022-08-31 14:38:15,744] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 14:38:15,745] [    INFO][0m -   Num examples = 12241[0m
[32m[2022-08-31 14:38:15,745] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 14:38:15,745] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 14:38:15,745] [    INFO][0m -   Total prediction steps = 192[0m
[32m[2022-08-31 14:39:10,698] [    INFO][0m - eval_loss: 0.5951129198074341, eval_accuracy: 0.7652152601911608, eval_runtime: 54.9529, eval_samples_per_second: 222.754, eval_steps_per_second: 3.494, epoch: 0.1143[0m
[32m[2022-08-31 14:39:10,699] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-700[0m
[32m[2022-08-31 14:39:10,699] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 14:39:12,459] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-700/tokenizer_config.json[0m
[32m[2022-08-31 14:39:12,459] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-700/special_tokens_map.json[0m
[32m[2022-08-31 14:39:21,243] [    INFO][0m - loss: 0.65470495, learning_rate: 2.9930414897092454e-05, global_step: 710, interval_runtime: 65.4999, interval_samples_per_second: 0.977, interval_steps_per_second: 0.153, epoch: 0.116[0m
[32m[2022-08-31 14:39:26,921] [    INFO][0m - loss: 0.62275553, learning_rate: 2.992943482522052e-05, global_step: 720, interval_runtime: 5.6777, interval_samples_per_second: 11.272, interval_steps_per_second: 1.761, epoch: 0.1176[0m
[32m[2022-08-31 14:39:32,566] [    INFO][0m - loss: 0.62941651, learning_rate: 2.992845475334858e-05, global_step: 730, interval_runtime: 5.6442, interval_samples_per_second: 11.339, interval_steps_per_second: 1.772, epoch: 0.1192[0m
[32m[2022-08-31 14:39:38,215] [    INFO][0m - loss: 0.66768112, learning_rate: 2.992747468147664e-05, global_step: 740, interval_runtime: 5.65, interval_samples_per_second: 11.327, interval_steps_per_second: 1.77, epoch: 0.1209[0m
[32m[2022-08-31 14:39:43,866] [    INFO][0m - loss: 0.67562971, learning_rate: 2.9926494609604705e-05, global_step: 750, interval_runtime: 5.6512, interval_samples_per_second: 11.325, interval_steps_per_second: 1.77, epoch: 0.1225[0m
[32m[2022-08-31 14:39:49,528] [    INFO][0m - loss: 0.69885988, learning_rate: 2.9925514537732766e-05, global_step: 760, interval_runtime: 5.6618, interval_samples_per_second: 11.304, interval_steps_per_second: 1.766, epoch: 0.1241[0m
[32m[2022-08-31 14:39:55,177] [    INFO][0m - loss: 0.66681585, learning_rate: 2.992453446586083e-05, global_step: 770, interval_runtime: 5.6487, interval_samples_per_second: 11.33, interval_steps_per_second: 1.77, epoch: 0.1258[0m
[32m[2022-08-31 14:40:00,816] [    INFO][0m - loss: 0.65928564, learning_rate: 2.992355439398889e-05, global_step: 780, interval_runtime: 5.6389, interval_samples_per_second: 11.35, interval_steps_per_second: 1.773, epoch: 0.1274[0m
[32m[2022-08-31 14:40:06,511] [    INFO][0m - loss: 0.63385005, learning_rate: 2.9922574322116956e-05, global_step: 790, interval_runtime: 5.6945, interval_samples_per_second: 11.239, interval_steps_per_second: 1.756, epoch: 0.129[0m
[32m[2022-08-31 14:40:12,210] [    INFO][0m - loss: 0.63931422, learning_rate: 2.9921594250245017e-05, global_step: 800, interval_runtime: 5.6997, interval_samples_per_second: 11.229, interval_steps_per_second: 1.754, epoch: 0.1307[0m
[32m[2022-08-31 14:40:12,211] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 14:40:12,211] [    INFO][0m -   Num examples = 12241[0m
[32m[2022-08-31 14:40:12,211] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 14:40:12,211] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 14:40:12,211] [    INFO][0m -   Total prediction steps = 192[0m
[32m[2022-08-31 14:41:06,681] [    INFO][0m - eval_loss: 0.5873503088951111, eval_accuracy: 0.7670124989788416, eval_runtime: 54.4698, eval_samples_per_second: 224.73, eval_steps_per_second: 3.525, epoch: 0.1307[0m
[32m[2022-08-31 14:41:06,682] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-800[0m
[32m[2022-08-31 14:41:06,682] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 14:41:08,502] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-800/tokenizer_config.json[0m
[32m[2022-08-31 14:41:08,502] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-800/special_tokens_map.json[0m
[32m[2022-08-31 14:41:17,214] [    INFO][0m - loss: 0.64155345, learning_rate: 2.992061417837308e-05, global_step: 810, interval_runtime: 65.0039, interval_samples_per_second: 0.985, interval_steps_per_second: 0.154, epoch: 0.1323[0m
[32m[2022-08-31 14:41:27,375] [    INFO][0m - loss: 0.63354478, learning_rate: 2.9919634106501142e-05, global_step: 820, interval_runtime: 5.6406, interval_samples_per_second: 11.346, interval_steps_per_second: 1.773, epoch: 0.1339[0m
[32m[2022-08-31 14:41:33,021] [    INFO][0m - loss: 0.63472128, learning_rate: 2.9918654034629207e-05, global_step: 830, interval_runtime: 10.1666, interval_samples_per_second: 6.295, interval_steps_per_second: 0.984, epoch: 0.1356[0m
[32m[2022-08-31 14:41:38,772] [    INFO][0m - loss: 0.70863099, learning_rate: 2.9917673962757268e-05, global_step: 840, interval_runtime: 5.7503, interval_samples_per_second: 11.13, interval_steps_per_second: 1.739, epoch: 0.1372[0m
[32m[2022-08-31 14:41:44,434] [    INFO][0m - loss: 0.60866089, learning_rate: 2.9916693890885332e-05, global_step: 850, interval_runtime: 5.6622, interval_samples_per_second: 11.303, interval_steps_per_second: 1.766, epoch: 0.1388[0m
[32m[2022-08-31 14:41:50,097] [    INFO][0m - loss: 0.62426796, learning_rate: 2.9915713819013397e-05, global_step: 860, interval_runtime: 5.6631, interval_samples_per_second: 11.301, interval_steps_per_second: 1.766, epoch: 0.1405[0m
[32m[2022-08-31 14:41:55,795] [    INFO][0m - loss: 0.65982299, learning_rate: 2.9914733747141458e-05, global_step: 870, interval_runtime: 5.6983, interval_samples_per_second: 11.232, interval_steps_per_second: 1.755, epoch: 0.1421[0m
[32m[2022-08-31 14:42:01,481] [    INFO][0m - loss: 0.66196623, learning_rate: 2.9913753675269522e-05, global_step: 880, interval_runtime: 5.6862, interval_samples_per_second: 11.255, interval_steps_per_second: 1.759, epoch: 0.1437[0m
[32m[2022-08-31 14:42:07,164] [    INFO][0m - loss: 0.58697653, learning_rate: 2.9912773603397583e-05, global_step: 890, interval_runtime: 5.6829, interval_samples_per_second: 11.262, interval_steps_per_second: 1.76, epoch: 0.1454[0m
[32m[2022-08-31 14:42:12,860] [    INFO][0m - loss: 0.67282152, learning_rate: 2.9911793531525647e-05, global_step: 900, interval_runtime: 5.6956, interval_samples_per_second: 11.237, interval_steps_per_second: 1.756, epoch: 0.147[0m
[32m[2022-08-31 14:42:12,860] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 14:42:12,860] [    INFO][0m -   Num examples = 12241[0m
[32m[2022-08-31 14:42:12,860] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 14:42:12,861] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 14:42:12,861] [    INFO][0m -   Total prediction steps = 192[0m
[32m[2022-08-31 14:43:09,061] [    INFO][0m - eval_loss: 0.565311074256897, eval_accuracy: 0.7747733028347358, eval_runtime: 56.1998, eval_samples_per_second: 217.812, eval_steps_per_second: 3.416, epoch: 0.147[0m
[32m[2022-08-31 14:43:09,061] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-900[0m
[32m[2022-08-31 14:43:09,062] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 14:43:10,916] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-900/tokenizer_config.json[0m
[32m[2022-08-31 14:43:10,916] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-900/special_tokens_map.json[0m
[32m[2022-08-31 14:43:19,679] [    INFO][0m - loss: 0.62082119, learning_rate: 2.991081345965371e-05, global_step: 910, interval_runtime: 66.8193, interval_samples_per_second: 0.958, interval_steps_per_second: 0.15, epoch: 0.1486[0m
[32m[2022-08-31 14:43:25,333] [    INFO][0m - loss: 0.61799698, learning_rate: 2.9909833387781773e-05, global_step: 920, interval_runtime: 5.6536, interval_samples_per_second: 11.32, interval_steps_per_second: 1.769, epoch: 0.1503[0m
[32m[2022-08-31 14:43:30,995] [    INFO][0m - loss: 0.63337731, learning_rate: 2.9908853315909834e-05, global_step: 930, interval_runtime: 5.6623, interval_samples_per_second: 11.303, interval_steps_per_second: 1.766, epoch: 0.1519[0m
[32m[2022-08-31 14:43:36,666] [    INFO][0m - loss: 0.61358442, learning_rate: 2.9907873244037898e-05, global_step: 940, interval_runtime: 5.6707, interval_samples_per_second: 11.286, interval_steps_per_second: 1.763, epoch: 0.1535[0m
[32m[2022-08-31 14:43:42,380] [    INFO][0m - loss: 0.6391098, learning_rate: 2.990689317216596e-05, global_step: 950, interval_runtime: 5.7139, interval_samples_per_second: 11.201, interval_steps_per_second: 1.75, epoch: 0.1552[0m
[32m[2022-08-31 14:43:48,069] [    INFO][0m - loss: 0.65804024, learning_rate: 2.9905913100294024e-05, global_step: 960, interval_runtime: 5.6891, interval_samples_per_second: 11.25, interval_steps_per_second: 1.758, epoch: 0.1568[0m
[32m[2022-08-31 14:43:53,771] [    INFO][0m - loss: 0.58952327, learning_rate: 2.9904933028422085e-05, global_step: 970, interval_runtime: 5.7016, interval_samples_per_second: 11.225, interval_steps_per_second: 1.754, epoch: 0.1584[0m
[32m[2022-08-31 14:43:59,451] [    INFO][0m - loss: 0.62964005, learning_rate: 2.990395295655015e-05, global_step: 980, interval_runtime: 5.6807, interval_samples_per_second: 11.266, interval_steps_per_second: 1.76, epoch: 0.1601[0m
[32m[2022-08-31 14:44:05,110] [    INFO][0m - loss: 0.61271086, learning_rate: 2.990297288467821e-05, global_step: 990, interval_runtime: 5.6594, interval_samples_per_second: 11.309, interval_steps_per_second: 1.767, epoch: 0.1617[0m
[32m[2022-08-31 14:44:10,775] [    INFO][0m - loss: 0.56349554, learning_rate: 2.9901992812806275e-05, global_step: 1000, interval_runtime: 5.6645, interval_samples_per_second: 11.298, interval_steps_per_second: 1.765, epoch: 0.1633[0m
[32m[2022-08-31 14:44:10,776] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 14:44:10,776] [    INFO][0m -   Num examples = 12241[0m
[32m[2022-08-31 14:44:10,776] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 14:44:10,776] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 14:44:10,776] [    INFO][0m -   Total prediction steps = 192[0m
[32m[2022-08-31 14:45:05,839] [    INFO][0m - eval_loss: 0.5801146030426025, eval_accuracy: 0.7701985131933665, eval_runtime: 55.0629, eval_samples_per_second: 222.309, eval_steps_per_second: 3.487, epoch: 0.1633[0m
[32m[2022-08-31 14:45:05,840] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-1000[0m
[32m[2022-08-31 14:45:05,840] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 14:45:07,819] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-1000/tokenizer_config.json[0m
[32m[2022-08-31 14:45:07,820] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-1000/special_tokens_map.json[0m
[32m[2022-08-31 14:45:16,693] [    INFO][0m - loss: 0.58439713, learning_rate: 2.9901012740934336e-05, global_step: 1010, interval_runtime: 65.9177, interval_samples_per_second: 0.971, interval_steps_per_second: 0.152, epoch: 0.165[0m
[32m[2022-08-31 14:45:22,355] [    INFO][0m - loss: 0.61495967, learning_rate: 2.9900032669062397e-05, global_step: 1020, interval_runtime: 5.6625, interval_samples_per_second: 11.302, interval_steps_per_second: 1.766, epoch: 0.1666[0m
[32m[2022-08-31 14:45:28,003] [    INFO][0m - loss: 0.60535965, learning_rate: 2.989905259719046e-05, global_step: 1030, interval_runtime: 5.6474, interval_samples_per_second: 11.333, interval_steps_per_second: 1.771, epoch: 0.1682[0m
[32m[2022-08-31 14:45:33,706] [    INFO][0m - loss: 0.56915951, learning_rate: 2.9898072525318522e-05, global_step: 1040, interval_runtime: 5.7035, interval_samples_per_second: 11.221, interval_steps_per_second: 1.753, epoch: 0.1699[0m
[32m[2022-08-31 14:45:39,313] [    INFO][0m - loss: 0.62581849, learning_rate: 2.9897092453446586e-05, global_step: 1050, interval_runtime: 5.6073, interval_samples_per_second: 11.414, interval_steps_per_second: 1.783, epoch: 0.1715[0m
[32m[2022-08-31 14:45:44,982] [    INFO][0m - loss: 0.61943359, learning_rate: 2.9896112381574647e-05, global_step: 1060, interval_runtime: 5.6682, interval_samples_per_second: 11.291, interval_steps_per_second: 1.764, epoch: 0.1731[0m
[32m[2022-08-31 14:45:50,674] [    INFO][0m - loss: 0.60002651, learning_rate: 2.9895132309702712e-05, global_step: 1070, interval_runtime: 5.6928, interval_samples_per_second: 11.242, interval_steps_per_second: 1.757, epoch: 0.1748[0m
[32m[2022-08-31 14:45:56,365] [    INFO][0m - loss: 0.65610366, learning_rate: 2.9894152237830773e-05, global_step: 1080, interval_runtime: 5.6905, interval_samples_per_second: 11.247, interval_steps_per_second: 1.757, epoch: 0.1764[0m
[32m[2022-08-31 14:46:02,025] [    INFO][0m - loss: 0.64170027, learning_rate: 2.9893172165958837e-05, global_step: 1090, interval_runtime: 5.6603, interval_samples_per_second: 11.307, interval_steps_per_second: 1.767, epoch: 0.178[0m
[32m[2022-08-31 14:46:07,691] [    INFO][0m - loss: 0.55998068, learning_rate: 2.9892192094086898e-05, global_step: 1100, interval_runtime: 5.6647, interval_samples_per_second: 11.298, interval_steps_per_second: 1.765, epoch: 0.1797[0m
[32m[2022-08-31 14:46:07,692] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 14:46:07,692] [    INFO][0m -   Num examples = 12241[0m
[32m[2022-08-31 14:46:07,692] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 14:46:07,692] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 14:46:07,692] [    INFO][0m -   Total prediction steps = 192[0m
[32m[2022-08-31 14:47:02,883] [    INFO][0m - eval_loss: 0.6170689463615417, eval_accuracy: 0.7605587778776244, eval_runtime: 55.1905, eval_samples_per_second: 221.796, eval_steps_per_second: 3.479, epoch: 0.1797[0m
[32m[2022-08-31 14:47:02,884] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-1100[0m
[32m[2022-08-31 14:47:02,884] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 14:47:04,970] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-1100/tokenizer_config.json[0m
[32m[2022-08-31 14:47:04,971] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-1100/special_tokens_map.json[0m
[32m[2022-08-31 14:47:13,868] [    INFO][0m - loss: 0.62611117, learning_rate: 2.9891212022214966e-05, global_step: 1110, interval_runtime: 66.1781, interval_samples_per_second: 0.967, interval_steps_per_second: 0.151, epoch: 0.1813[0m
[32m[2022-08-31 14:47:19,497] [    INFO][0m - loss: 0.58900542, learning_rate: 2.9890231950343027e-05, global_step: 1120, interval_runtime: 5.6292, interval_samples_per_second: 11.369, interval_steps_per_second: 1.776, epoch: 0.1829[0m
[32m[2022-08-31 14:47:27,754] [    INFO][0m - loss: 0.63815007, learning_rate: 2.988925187847109e-05, global_step: 1130, interval_runtime: 5.6449, interval_samples_per_second: 11.338, interval_steps_per_second: 1.772, epoch: 0.1846[0m
[32m[2022-08-31 14:47:33,394] [    INFO][0m - loss: 0.62994857, learning_rate: 2.9888271806599152e-05, global_step: 1140, interval_runtime: 8.2514, interval_samples_per_second: 7.756, interval_steps_per_second: 1.212, epoch: 0.1862[0m
[32m[2022-08-31 14:47:39,015] [    INFO][0m - loss: 0.6357286, learning_rate: 2.9887291734727217e-05, global_step: 1150, interval_runtime: 5.6211, interval_samples_per_second: 11.386, interval_steps_per_second: 1.779, epoch: 0.1878[0m
[32m[2022-08-31 14:47:44,658] [    INFO][0m - loss: 0.62469001, learning_rate: 2.9886311662855278e-05, global_step: 1160, interval_runtime: 5.6437, interval_samples_per_second: 11.34, interval_steps_per_second: 1.772, epoch: 0.1895[0m
[32m[2022-08-31 14:47:50,331] [    INFO][0m - loss: 0.58288693, learning_rate: 2.988533159098334e-05, global_step: 1170, interval_runtime: 5.6726, interval_samples_per_second: 11.282, interval_steps_per_second: 1.763, epoch: 0.1911[0m
[32m[2022-08-31 14:47:55,967] [    INFO][0m - loss: 0.65038319, learning_rate: 2.9884351519111403e-05, global_step: 1180, interval_runtime: 5.6358, interval_samples_per_second: 11.356, interval_steps_per_second: 1.774, epoch: 0.1927[0m
[32m[2022-08-31 14:48:01,629] [    INFO][0m - loss: 0.57681084, learning_rate: 2.9883371447239464e-05, global_step: 1190, interval_runtime: 5.662, interval_samples_per_second: 11.303, interval_steps_per_second: 1.766, epoch: 0.1944[0m
[32m[2022-08-31 14:48:07,329] [    INFO][0m - loss: 0.5447114, learning_rate: 2.988239137536753e-05, global_step: 1200, interval_runtime: 5.7002, interval_samples_per_second: 11.228, interval_steps_per_second: 1.754, epoch: 0.196[0m
[32m[2022-08-31 14:48:07,330] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 14:48:07,330] [    INFO][0m -   Num examples = 12241[0m
[32m[2022-08-31 14:48:07,330] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 14:48:07,330] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 14:48:07,330] [    INFO][0m -   Total prediction steps = 192[0m
[32m[2022-08-31 14:49:02,168] [    INFO][0m - eval_loss: 0.555372416973114, eval_accuracy: 0.7823707213462953, eval_runtime: 54.8375, eval_samples_per_second: 223.223, eval_steps_per_second: 3.501, epoch: 0.196[0m
[32m[2022-08-31 14:49:02,168] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-1200[0m
[32m[2022-08-31 14:49:02,169] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 14:49:04,230] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-1200/tokenizer_config.json[0m
[32m[2022-08-31 14:49:04,230] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-1200/special_tokens_map.json[0m
[32m[2022-08-31 14:49:13,121] [    INFO][0m - loss: 0.60508404, learning_rate: 2.988141130349559e-05, global_step: 1210, interval_runtime: 65.7921, interval_samples_per_second: 0.973, interval_steps_per_second: 0.152, epoch: 0.1976[0m
[32m[2022-08-31 14:49:18,748] [    INFO][0m - loss: 0.57423673, learning_rate: 2.9880431231623654e-05, global_step: 1220, interval_runtime: 5.6266, interval_samples_per_second: 11.375, interval_steps_per_second: 1.777, epoch: 0.1993[0m
[32m[2022-08-31 14:49:24,409] [    INFO][0m - loss: 0.60501599, learning_rate: 2.9879451159751715e-05, global_step: 1230, interval_runtime: 5.6612, interval_samples_per_second: 11.305, interval_steps_per_second: 1.766, epoch: 0.2009[0m
[32m[2022-08-31 14:49:30,057] [    INFO][0m - loss: 0.61042557, learning_rate: 2.987847108787978e-05, global_step: 1240, interval_runtime: 5.6476, interval_samples_per_second: 11.332, interval_steps_per_second: 1.771, epoch: 0.2025[0m
[32m[2022-08-31 14:49:35,692] [    INFO][0m - loss: 0.63335381, learning_rate: 2.987749101600784e-05, global_step: 1250, interval_runtime: 5.6353, interval_samples_per_second: 11.357, interval_steps_per_second: 1.775, epoch: 0.2042[0m
[32m[2022-08-31 14:49:41,426] [    INFO][0m - loss: 0.61361933, learning_rate: 2.9876510944135905e-05, global_step: 1260, interval_runtime: 5.7336, interval_samples_per_second: 11.162, interval_steps_per_second: 1.744, epoch: 0.2058[0m
[32m[2022-08-31 14:49:47,118] [    INFO][0m - loss: 0.62543736, learning_rate: 2.9875530872263966e-05, global_step: 1270, interval_runtime: 5.6919, interval_samples_per_second: 11.244, interval_steps_per_second: 1.757, epoch: 0.2074[0m
[32m[2022-08-31 14:49:52,785] [    INFO][0m - loss: 0.6286622, learning_rate: 2.987455080039203e-05, global_step: 1280, interval_runtime: 5.6674, interval_samples_per_second: 11.293, interval_steps_per_second: 1.764, epoch: 0.2091[0m
[32m[2022-08-31 14:49:58,447] [    INFO][0m - loss: 0.64121571, learning_rate: 2.987357072852009e-05, global_step: 1290, interval_runtime: 5.6618, interval_samples_per_second: 11.304, interval_steps_per_second: 1.766, epoch: 0.2107[0m
[32m[2022-08-31 14:50:04,093] [    INFO][0m - loss: 0.57496786, learning_rate: 2.9872590656648156e-05, global_step: 1300, interval_runtime: 5.6463, interval_samples_per_second: 11.335, interval_steps_per_second: 1.771, epoch: 0.2123[0m
[32m[2022-08-31 14:50:04,094] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 14:50:04,094] [    INFO][0m -   Num examples = 12241[0m
[32m[2022-08-31 14:50:04,094] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 14:50:04,094] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 14:50:04,094] [    INFO][0m -   Total prediction steps = 192[0m
[32m[2022-08-31 14:50:58,883] [    INFO][0m - eval_loss: 0.5530857443809509, eval_accuracy: 0.7794297851482722, eval_runtime: 54.7887, eval_samples_per_second: 223.422, eval_steps_per_second: 3.504, epoch: 0.2123[0m
[32m[2022-08-31 14:50:58,884] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-1300[0m
[32m[2022-08-31 14:50:58,884] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 14:51:00,982] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-1300/tokenizer_config.json[0m
[32m[2022-08-31 14:51:00,982] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-1300/special_tokens_map.json[0m
[32m[2022-08-31 14:51:09,888] [    INFO][0m - loss: 0.55333352, learning_rate: 2.9871610584776217e-05, global_step: 1310, interval_runtime: 65.7954, interval_samples_per_second: 0.973, interval_steps_per_second: 0.152, epoch: 0.214[0m
[32m[2022-08-31 14:51:15,528] [    INFO][0m - loss: 0.61171017, learning_rate: 2.9870630512904278e-05, global_step: 1320, interval_runtime: 5.6395, interval_samples_per_second: 11.349, interval_steps_per_second: 1.773, epoch: 0.2156[0m
[32m[2022-08-31 14:51:21,245] [    INFO][0m - loss: 0.60487995, learning_rate: 2.9869650441032342e-05, global_step: 1330, interval_runtime: 5.7171, interval_samples_per_second: 11.194, interval_steps_per_second: 1.749, epoch: 0.2172[0m
[32m[2022-08-31 14:51:26,882] [    INFO][0m - loss: 0.57937732, learning_rate: 2.9868670369160403e-05, global_step: 1340, interval_runtime: 5.6364, interval_samples_per_second: 11.355, interval_steps_per_second: 1.774, epoch: 0.2189[0m
[32m[2022-08-31 14:51:32,568] [    INFO][0m - loss: 0.60604949, learning_rate: 2.9867690297288467e-05, global_step: 1350, interval_runtime: 5.6861, interval_samples_per_second: 11.255, interval_steps_per_second: 1.759, epoch: 0.2205[0m
[32m[2022-08-31 14:51:38,223] [    INFO][0m - loss: 0.60687752, learning_rate: 2.9866710225416532e-05, global_step: 1360, interval_runtime: 5.6557, interval_samples_per_second: 11.316, interval_steps_per_second: 1.768, epoch: 0.2221[0m
[32m[2022-08-31 14:51:43,895] [    INFO][0m - loss: 0.62404137, learning_rate: 2.9865730153544596e-05, global_step: 1370, interval_runtime: 5.6719, interval_samples_per_second: 11.284, interval_steps_per_second: 1.763, epoch: 0.2238[0m
[32m[2022-08-31 14:51:49,660] [    INFO][0m - loss: 0.57408199, learning_rate: 2.9864750081672657e-05, global_step: 1380, interval_runtime: 5.7642, interval_samples_per_second: 11.103, interval_steps_per_second: 1.735, epoch: 0.2254[0m
[32m[2022-08-31 14:51:55,298] [    INFO][0m - loss: 0.62745438, learning_rate: 2.986377000980072e-05, global_step: 1390, interval_runtime: 5.6386, interval_samples_per_second: 11.35, interval_steps_per_second: 1.774, epoch: 0.227[0m
[32m[2022-08-31 14:52:00,977] [    INFO][0m - loss: 0.54885035, learning_rate: 2.9862789937928783e-05, global_step: 1400, interval_runtime: 5.6788, interval_samples_per_second: 11.27, interval_steps_per_second: 1.761, epoch: 0.2287[0m
[32m[2022-08-31 14:52:00,977] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 14:52:00,977] [    INFO][0m -   Num examples = 12241[0m
[32m[2022-08-31 14:52:00,977] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 14:52:00,977] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 14:52:00,978] [    INFO][0m -   Total prediction steps = 192[0m
[32m[2022-08-31 14:52:56,420] [    INFO][0m - eval_loss: 0.553166389465332, eval_accuracy: 0.7823707213462953, eval_runtime: 55.4423, eval_samples_per_second: 220.788, eval_steps_per_second: 3.463, epoch: 0.2287[0m
[32m[2022-08-31 14:52:56,421] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-1400[0m
[32m[2022-08-31 14:52:56,421] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 14:52:58,297] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-1400/tokenizer_config.json[0m
[32m[2022-08-31 14:52:58,297] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-1400/special_tokens_map.json[0m
[32m[2022-08-31 14:53:06,826] [    INFO][0m - loss: 0.61843286, learning_rate: 2.9861809866056847e-05, global_step: 1410, interval_runtime: 65.8497, interval_samples_per_second: 0.972, interval_steps_per_second: 0.152, epoch: 0.2303[0m
[32m[2022-08-31 14:53:12,453] [    INFO][0m - loss: 0.65639539, learning_rate: 2.9860829794184908e-05, global_step: 1420, interval_runtime: 5.6269, interval_samples_per_second: 11.374, interval_steps_per_second: 1.777, epoch: 0.232[0m
[32m[2022-08-31 14:53:18,125] [    INFO][0m - loss: 0.57854066, learning_rate: 2.9859849722312972e-05, global_step: 1430, interval_runtime: 5.6716, interval_samples_per_second: 11.284, interval_steps_per_second: 1.763, epoch: 0.2336[0m
[32m[2022-08-31 14:53:23,820] [    INFO][0m - loss: 0.61603436, learning_rate: 2.9858869650441033e-05, global_step: 1440, interval_runtime: 5.6946, interval_samples_per_second: 11.239, interval_steps_per_second: 1.756, epoch: 0.2352[0m
[32m[2022-08-31 14:53:29,475] [    INFO][0m - loss: 0.57850389, learning_rate: 2.9857889578569094e-05, global_step: 1450, interval_runtime: 5.6554, interval_samples_per_second: 11.317, interval_steps_per_second: 1.768, epoch: 0.2369[0m
[32m[2022-08-31 14:53:35,152] [    INFO][0m - loss: 0.56365514, learning_rate: 2.985690950669716e-05, global_step: 1460, interval_runtime: 5.6768, interval_samples_per_second: 11.274, interval_steps_per_second: 1.762, epoch: 0.2385[0m
[32m[2022-08-31 14:53:41,558] [    INFO][0m - loss: 0.55920401, learning_rate: 2.985592943482522e-05, global_step: 1470, interval_runtime: 5.6581, interval_samples_per_second: 11.311, interval_steps_per_second: 1.767, epoch: 0.2401[0m
[32m[2022-08-31 14:53:47,238] [    INFO][0m - loss: 0.61596832, learning_rate: 2.9854949362953284e-05, global_step: 1480, interval_runtime: 6.4281, interval_samples_per_second: 9.956, interval_steps_per_second: 1.556, epoch: 0.2418[0m
[32m[2022-08-31 14:53:52,904] [    INFO][0m - loss: 0.62025356, learning_rate: 2.9853969291081345e-05, global_step: 1490, interval_runtime: 5.6662, interval_samples_per_second: 11.295, interval_steps_per_second: 1.765, epoch: 0.2434[0m
[32m[2022-08-31 14:53:58,574] [    INFO][0m - loss: 0.59076791, learning_rate: 2.985298921920941e-05, global_step: 1500, interval_runtime: 5.6694, interval_samples_per_second: 11.289, interval_steps_per_second: 1.764, epoch: 0.245[0m
[32m[2022-08-31 14:53:58,574] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 14:53:58,574] [    INFO][0m -   Num examples = 12241[0m
[32m[2022-08-31 14:53:58,574] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 14:53:58,574] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 14:53:58,575] [    INFO][0m -   Total prediction steps = 192[0m
[32m[2022-08-31 14:54:53,286] [    INFO][0m - eval_loss: 0.5360491275787354, eval_accuracy: 0.7835144187566375, eval_runtime: 54.7115, eval_samples_per_second: 223.737, eval_steps_per_second: 3.509, epoch: 0.245[0m
[32m[2022-08-31 14:54:53,287] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-1500[0m
[32m[2022-08-31 14:54:53,287] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 14:54:54,857] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-1500/tokenizer_config.json[0m
[32m[2022-08-31 14:54:54,857] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-1500/special_tokens_map.json[0m
[32m[2022-08-31 14:55:03,081] [    INFO][0m - loss: 0.56451049, learning_rate: 2.985200914733747e-05, global_step: 1510, interval_runtime: 64.5069, interval_samples_per_second: 0.992, interval_steps_per_second: 0.155, epoch: 0.2467[0m
[32m[2022-08-31 14:55:08,714] [    INFO][0m - loss: 0.57734356, learning_rate: 2.9851029075465535e-05, global_step: 1520, interval_runtime: 5.6331, interval_samples_per_second: 11.361, interval_steps_per_second: 1.775, epoch: 0.2483[0m
[32m[2022-08-31 14:55:14,414] [    INFO][0m - loss: 0.57877998, learning_rate: 2.9850049003593596e-05, global_step: 1530, interval_runtime: 5.7005, interval_samples_per_second: 11.227, interval_steps_per_second: 1.754, epoch: 0.2499[0m
[32m[2022-08-31 14:55:20,061] [    INFO][0m - loss: 0.62211113, learning_rate: 2.984906893172166e-05, global_step: 1540, interval_runtime: 5.6462, interval_samples_per_second: 11.335, interval_steps_per_second: 1.771, epoch: 0.2516[0m
[32m[2022-08-31 14:55:25,703] [    INFO][0m - loss: 0.56717286, learning_rate: 2.984808885984972e-05, global_step: 1550, interval_runtime: 5.6425, interval_samples_per_second: 11.342, interval_steps_per_second: 1.772, epoch: 0.2532[0m
[32m[2022-08-31 14:55:31,445] [    INFO][0m - loss: 0.61293755, learning_rate: 2.9847108787977786e-05, global_step: 1560, interval_runtime: 5.7417, interval_samples_per_second: 11.146, interval_steps_per_second: 1.742, epoch: 0.2548[0m
[32m[2022-08-31 14:55:37,103] [    INFO][0m - loss: 0.59084187, learning_rate: 2.9846128716105847e-05, global_step: 1570, interval_runtime: 5.6584, interval_samples_per_second: 11.311, interval_steps_per_second: 1.767, epoch: 0.2565[0m
[32m[2022-08-31 14:55:42,768] [    INFO][0m - loss: 0.62277107, learning_rate: 2.984514864423391e-05, global_step: 1580, interval_runtime: 5.6653, interval_samples_per_second: 11.297, interval_steps_per_second: 1.765, epoch: 0.2581[0m
[32m[2022-08-31 14:55:48,411] [    INFO][0m - loss: 0.57373343, learning_rate: 2.9844168572361972e-05, global_step: 1590, interval_runtime: 5.6429, interval_samples_per_second: 11.342, interval_steps_per_second: 1.772, epoch: 0.2597[0m
[32m[2022-08-31 14:55:54,067] [    INFO][0m - loss: 0.62317452, learning_rate: 2.9843188500490037e-05, global_step: 1600, interval_runtime: 5.6555, interval_samples_per_second: 11.316, interval_steps_per_second: 1.768, epoch: 0.2614[0m
[32m[2022-08-31 14:55:54,068] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 14:55:54,068] [    INFO][0m -   Num examples = 12241[0m
[32m[2022-08-31 14:55:54,068] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 14:55:54,068] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 14:55:54,068] [    INFO][0m -   Total prediction steps = 192[0m
[32m[2022-08-31 14:56:49,186] [    INFO][0m - eval_loss: 0.5297749042510986, eval_accuracy: 0.7837594967731395, eval_runtime: 55.1171, eval_samples_per_second: 222.091, eval_steps_per_second: 3.483, epoch: 0.2614[0m
[32m[2022-08-31 14:56:49,186] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-1600[0m
[32m[2022-08-31 14:56:49,186] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 14:56:50,921] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-1600/tokenizer_config.json[0m
[32m[2022-08-31 14:56:50,922] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-1600/special_tokens_map.json[0m
[32m[2022-08-31 14:56:59,571] [    INFO][0m - loss: 0.58094192, learning_rate: 2.98422084286181e-05, global_step: 1610, interval_runtime: 65.5036, interval_samples_per_second: 0.977, interval_steps_per_second: 0.153, epoch: 0.263[0m
[32m[2022-08-31 14:57:05,234] [    INFO][0m - loss: 0.59562221, learning_rate: 2.9841228356746162e-05, global_step: 1620, interval_runtime: 5.6631, interval_samples_per_second: 11.301, interval_steps_per_second: 1.766, epoch: 0.2646[0m
[32m[2022-08-31 14:57:10,904] [    INFO][0m - loss: 0.5434185, learning_rate: 2.9840248284874227e-05, global_step: 1630, interval_runtime: 5.6705, interval_samples_per_second: 11.286, interval_steps_per_second: 1.764, epoch: 0.2663[0m
[32m[2022-08-31 14:57:16,551] [    INFO][0m - loss: 0.57149715, learning_rate: 2.9839268213002288e-05, global_step: 1640, interval_runtime: 5.6471, interval_samples_per_second: 11.333, interval_steps_per_second: 1.771, epoch: 0.2679[0m
[32m[2022-08-31 14:57:22,236] [    INFO][0m - loss: 0.58673687, learning_rate: 2.9838288141130352e-05, global_step: 1650, interval_runtime: 5.6845, interval_samples_per_second: 11.259, interval_steps_per_second: 1.759, epoch: 0.2695[0m
[32m[2022-08-31 14:57:27,894] [    INFO][0m - loss: 0.62249317, learning_rate: 2.9837308069258413e-05, global_step: 1660, interval_runtime: 5.6578, interval_samples_per_second: 11.312, interval_steps_per_second: 1.767, epoch: 0.2712[0m
[32m[2022-08-31 14:57:33,543] [    INFO][0m - loss: 0.62195778, learning_rate: 2.9836327997386477e-05, global_step: 1670, interval_runtime: 5.6491, interval_samples_per_second: 11.329, interval_steps_per_second: 1.77, epoch: 0.2728[0m
[32m[2022-08-31 14:57:39,213] [    INFO][0m - loss: 0.64759045, learning_rate: 2.983534792551454e-05, global_step: 1680, interval_runtime: 5.6705, interval_samples_per_second: 11.287, interval_steps_per_second: 1.764, epoch: 0.2744[0m
[32m[2022-08-31 14:57:44,863] [    INFO][0m - loss: 0.58314495, learning_rate: 2.9834367853642603e-05, global_step: 1690, interval_runtime: 5.6494, interval_samples_per_second: 11.329, interval_steps_per_second: 1.77, epoch: 0.2761[0m
[32m[2022-08-31 14:57:50,518] [    INFO][0m - loss: 0.58546643, learning_rate: 2.9833387781770664e-05, global_step: 1700, interval_runtime: 5.6558, interval_samples_per_second: 11.316, interval_steps_per_second: 1.768, epoch: 0.2777[0m
[32m[2022-08-31 14:57:50,519] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 14:57:50,519] [    INFO][0m -   Num examples = 12241[0m
[32m[2022-08-31 14:57:50,519] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 14:57:50,519] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 14:57:50,519] [    INFO][0m -   Total prediction steps = 192[0m
[32m[2022-08-31 14:58:45,686] [    INFO][0m - eval_loss: 0.5367875695228577, eval_accuracy: 0.7875990523650028, eval_runtime: 55.1666, eval_samples_per_second: 221.892, eval_steps_per_second: 3.48, epoch: 0.2777[0m
[32m[2022-08-31 14:58:45,687] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-1700[0m
[32m[2022-08-31 14:58:45,687] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 14:58:47,483] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-1700/tokenizer_config.json[0m
[32m[2022-08-31 14:58:47,484] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-1700/special_tokens_map.json[0m
[32m[2022-08-31 14:58:56,082] [    INFO][0m - loss: 0.59190412, learning_rate: 2.9832407709898728e-05, global_step: 1710, interval_runtime: 65.5636, interval_samples_per_second: 0.976, interval_steps_per_second: 0.153, epoch: 0.2793[0m
[32m[2022-08-31 14:59:01,749] [    INFO][0m - loss: 0.60594511, learning_rate: 2.983142763802679e-05, global_step: 1720, interval_runtime: 5.6661, interval_samples_per_second: 11.295, interval_steps_per_second: 1.765, epoch: 0.281[0m
[32m[2022-08-31 14:59:07,379] [    INFO][0m - loss: 0.56993423, learning_rate: 2.9830447566154854e-05, global_step: 1730, interval_runtime: 5.6304, interval_samples_per_second: 11.367, interval_steps_per_second: 1.776, epoch: 0.2826[0m
[32m[2022-08-31 14:59:13,052] [    INFO][0m - loss: 0.59765821, learning_rate: 2.9829467494282915e-05, global_step: 1740, interval_runtime: 5.673, interval_samples_per_second: 11.281, interval_steps_per_second: 1.763, epoch: 0.2842[0m
[32m[2022-08-31 14:59:18,714] [    INFO][0m - loss: 0.59008164, learning_rate: 2.9828487422410976e-05, global_step: 1750, interval_runtime: 5.6618, interval_samples_per_second: 11.304, interval_steps_per_second: 1.766, epoch: 0.2859[0m
[32m[2022-08-31 14:59:24,416] [    INFO][0m - loss: 0.58763118, learning_rate: 2.982750735053904e-05, global_step: 1760, interval_runtime: 5.7023, interval_samples_per_second: 11.224, interval_steps_per_second: 1.754, epoch: 0.2875[0m
[32m[2022-08-31 14:59:30,086] [    INFO][0m - loss: 0.61279125, learning_rate: 2.98265272786671e-05, global_step: 1770, interval_runtime: 5.6705, interval_samples_per_second: 11.287, interval_steps_per_second: 1.764, epoch: 0.2891[0m
[32m[2022-08-31 14:59:35,750] [    INFO][0m - loss: 0.59157143, learning_rate: 2.9825547206795165e-05, global_step: 1780, interval_runtime: 5.6633, interval_samples_per_second: 11.301, interval_steps_per_second: 1.766, epoch: 0.2908[0m
[32m[2022-08-31 14:59:41,572] [    INFO][0m - loss: 0.60344167, learning_rate: 2.9824567134923226e-05, global_step: 1790, interval_runtime: 5.6879, interval_samples_per_second: 11.252, interval_steps_per_second: 1.758, epoch: 0.2924[0m
[32m[2022-08-31 14:59:47,222] [    INFO][0m - loss: 0.5403687, learning_rate: 2.982358706305129e-05, global_step: 1800, interval_runtime: 5.7845, interval_samples_per_second: 11.064, interval_steps_per_second: 1.729, epoch: 0.294[0m
[32m[2022-08-31 14:59:47,223] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 14:59:47,223] [    INFO][0m -   Num examples = 12241[0m
[32m[2022-08-31 14:59:47,223] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 14:59:47,223] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 14:59:47,223] [    INFO][0m -   Total prediction steps = 192[0m
[32m[2022-08-31 15:00:42,122] [    INFO][0m - eval_loss: 0.5278180241584778, eval_accuracy: 0.7938893881218855, eval_runtime: 54.8986, eval_samples_per_second: 222.975, eval_steps_per_second: 3.497, epoch: 0.294[0m
[32m[2022-08-31 15:00:42,123] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-1800[0m
[32m[2022-08-31 15:00:42,123] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:00:44,005] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-1800/tokenizer_config.json[0m
[32m[2022-08-31 15:00:44,005] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-1800/special_tokens_map.json[0m
[32m[2022-08-31 15:00:52,677] [    INFO][0m - loss: 0.60217543, learning_rate: 2.9822606991179352e-05, global_step: 1810, interval_runtime: 65.4548, interval_samples_per_second: 0.978, interval_steps_per_second: 0.153, epoch: 0.2957[0m
[32m[2022-08-31 15:00:58,365] [    INFO][0m - loss: 0.55327077, learning_rate: 2.9821626919307416e-05, global_step: 1820, interval_runtime: 5.6877, interval_samples_per_second: 11.252, interval_steps_per_second: 1.758, epoch: 0.2973[0m
[32m[2022-08-31 15:01:04,037] [    INFO][0m - loss: 0.55364919, learning_rate: 2.9820646847435477e-05, global_step: 1830, interval_runtime: 5.6724, interval_samples_per_second: 11.283, interval_steps_per_second: 1.763, epoch: 0.2989[0m
[32m[2022-08-31 15:01:09,709] [    INFO][0m - loss: 0.58890853, learning_rate: 2.981966677556354e-05, global_step: 1840, interval_runtime: 5.6716, interval_samples_per_second: 11.284, interval_steps_per_second: 1.763, epoch: 0.3006[0m
[32m[2022-08-31 15:01:15,381] [    INFO][0m - loss: 0.6124474, learning_rate: 2.9818686703691606e-05, global_step: 1850, interval_runtime: 5.6727, interval_samples_per_second: 11.282, interval_steps_per_second: 1.763, epoch: 0.3022[0m
[32m[2022-08-31 15:01:21,037] [    INFO][0m - loss: 0.56920991, learning_rate: 2.981770663181967e-05, global_step: 1860, interval_runtime: 5.6555, interval_samples_per_second: 11.316, interval_steps_per_second: 1.768, epoch: 0.3038[0m
[32m[2022-08-31 15:01:26,684] [    INFO][0m - loss: 0.57007346, learning_rate: 2.981672655994773e-05, global_step: 1870, interval_runtime: 5.647, interval_samples_per_second: 11.333, interval_steps_per_second: 1.771, epoch: 0.3055[0m
[32m[2022-08-31 15:01:32,333] [    INFO][0m - loss: 0.59314232, learning_rate: 2.9815746488075796e-05, global_step: 1880, interval_runtime: 5.649, interval_samples_per_second: 11.329, interval_steps_per_second: 1.77, epoch: 0.3071[0m
[32m[2022-08-31 15:01:37,970] [    INFO][0m - loss: 0.59272833, learning_rate: 2.9814766416203857e-05, global_step: 1890, interval_runtime: 5.637, interval_samples_per_second: 11.353, interval_steps_per_second: 1.774, epoch: 0.3087[0m
[32m[2022-08-31 15:01:43,621] [    INFO][0m - loss: 0.60170765, learning_rate: 2.9813786344331918e-05, global_step: 1900, interval_runtime: 5.6508, interval_samples_per_second: 11.326, interval_steps_per_second: 1.77, epoch: 0.3104[0m
[32m[2022-08-31 15:01:43,621] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 15:01:43,621] [    INFO][0m -   Num examples = 12241[0m
[32m[2022-08-31 15:01:43,621] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:01:43,621] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:01:43,621] [    INFO][0m -   Total prediction steps = 192[0m
[32m[2022-08-31 15:02:38,330] [    INFO][0m - eval_loss: 0.5282647013664246, eval_accuracy: 0.7908667592516951, eval_runtime: 54.7081, eval_samples_per_second: 223.751, eval_steps_per_second: 3.51, epoch: 0.3104[0m
[32m[2022-08-31 15:02:38,331] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-1900[0m
[32m[2022-08-31 15:02:38,331] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:02:39,993] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-1900/tokenizer_config.json[0m
[32m[2022-08-31 15:02:39,993] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-1900/special_tokens_map.json[0m
[32m[2022-08-31 15:02:48,210] [    INFO][0m - loss: 0.55691338, learning_rate: 2.9812806272459982e-05, global_step: 1910, interval_runtime: 64.5897, interval_samples_per_second: 0.991, interval_steps_per_second: 0.155, epoch: 0.312[0m
[32m[2022-08-31 15:02:53,855] [    INFO][0m - loss: 0.57359643, learning_rate: 2.9811826200588043e-05, global_step: 1920, interval_runtime: 5.6447, interval_samples_per_second: 11.338, interval_steps_per_second: 1.772, epoch: 0.3136[0m
[32m[2022-08-31 15:02:59,515] [    INFO][0m - loss: 0.56619396, learning_rate: 2.9810846128716108e-05, global_step: 1930, interval_runtime: 5.6595, interval_samples_per_second: 11.308, interval_steps_per_second: 1.767, epoch: 0.3153[0m
[32m[2022-08-31 15:03:05,200] [    INFO][0m - loss: 0.59350348, learning_rate: 2.980986605684417e-05, global_step: 1940, interval_runtime: 5.6852, interval_samples_per_second: 11.257, interval_steps_per_second: 1.759, epoch: 0.3169[0m
[32m[2022-08-31 15:03:10,848] [    INFO][0m - loss: 0.56815639, learning_rate: 2.9808885984972233e-05, global_step: 1950, interval_runtime: 5.648, interval_samples_per_second: 11.331, interval_steps_per_second: 1.771, epoch: 0.3185[0m
[32m[2022-08-31 15:03:16,498] [    INFO][0m - loss: 0.56388998, learning_rate: 2.9807905913100294e-05, global_step: 1960, interval_runtime: 5.6501, interval_samples_per_second: 11.327, interval_steps_per_second: 1.77, epoch: 0.3202[0m
[32m[2022-08-31 15:03:22,187] [    INFO][0m - loss: 0.48070574, learning_rate: 2.980692584122836e-05, global_step: 1970, interval_runtime: 5.6894, interval_samples_per_second: 11.249, interval_steps_per_second: 1.758, epoch: 0.3218[0m
[32m[2022-08-31 15:03:27,847] [    INFO][0m - loss: 0.60918369, learning_rate: 2.980594576935642e-05, global_step: 1980, interval_runtime: 5.6598, interval_samples_per_second: 11.308, interval_steps_per_second: 1.767, epoch: 0.3234[0m
[32m[2022-08-31 15:03:33,507] [    INFO][0m - loss: 0.57777853, learning_rate: 2.9804965697484484e-05, global_step: 1990, interval_runtime: 5.66, interval_samples_per_second: 11.307, interval_steps_per_second: 1.767, epoch: 0.3251[0m
[32m[2022-08-31 15:03:39,157] [    INFO][0m - loss: 0.5777235, learning_rate: 2.9803985625612545e-05, global_step: 2000, interval_runtime: 5.6499, interval_samples_per_second: 11.328, interval_steps_per_second: 1.77, epoch: 0.3267[0m
[32m[2022-08-31 15:03:39,158] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 15:03:39,158] [    INFO][0m -   Num examples = 12241[0m
[32m[2022-08-31 15:03:39,158] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:03:39,158] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:03:39,158] [    INFO][0m -   Total prediction steps = 192[0m
[32m[2022-08-31 15:04:34,131] [    INFO][0m - eval_loss: 0.55435711145401, eval_accuracy: 0.7826157993627971, eval_runtime: 54.9725, eval_samples_per_second: 222.675, eval_steps_per_second: 3.493, epoch: 0.3267[0m
[32m[2022-08-31 15:04:34,132] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-2000[0m
[32m[2022-08-31 15:04:34,132] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:04:35,721] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-2000/tokenizer_config.json[0m
[32m[2022-08-31 15:04:35,722] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-2000/special_tokens_map.json[0m
[32m[2022-08-31 15:04:43,842] [    INFO][0m - loss: 0.55722528, learning_rate: 2.980300555374061e-05, global_step: 2010, interval_runtime: 64.6842, interval_samples_per_second: 0.989, interval_steps_per_second: 0.155, epoch: 0.3283[0m
[32m[2022-08-31 15:04:49,515] [    INFO][0m - loss: 0.60586548, learning_rate: 2.980202548186867e-05, global_step: 2020, interval_runtime: 5.6731, interval_samples_per_second: 11.281, interval_steps_per_second: 1.763, epoch: 0.33[0m
[32m[2022-08-31 15:04:55,194] [    INFO][0m - loss: 0.55284314, learning_rate: 2.980104540999673e-05, global_step: 2030, interval_runtime: 5.679, interval_samples_per_second: 11.27, interval_steps_per_second: 1.761, epoch: 0.3316[0m
[32m[2022-08-31 15:05:00,830] [    INFO][0m - loss: 0.51296782, learning_rate: 2.9800065338124796e-05, global_step: 2040, interval_runtime: 5.6359, interval_samples_per_second: 11.356, interval_steps_per_second: 1.774, epoch: 0.3332[0m
[32m[2022-08-31 15:05:06,471] [    INFO][0m - loss: 0.64164891, learning_rate: 2.9799085266252857e-05, global_step: 2050, interval_runtime: 5.6411, interval_samples_per_second: 11.345, interval_steps_per_second: 1.773, epoch: 0.3349[0m
[32m[2022-08-31 15:05:12,140] [    INFO][0m - loss: 0.57461624, learning_rate: 2.979810519438092e-05, global_step: 2060, interval_runtime: 5.6696, interval_samples_per_second: 11.288, interval_steps_per_second: 1.764, epoch: 0.3365[0m
[32m[2022-08-31 15:05:17,798] [    INFO][0m - loss: 0.51923556, learning_rate: 2.9797125122508982e-05, global_step: 2070, interval_runtime: 5.6575, interval_samples_per_second: 11.313, interval_steps_per_second: 1.768, epoch: 0.3381[0m
[32m[2022-08-31 15:05:23,450] [    INFO][0m - loss: 0.61698031, learning_rate: 2.9796145050637046e-05, global_step: 2080, interval_runtime: 5.6517, interval_samples_per_second: 11.324, interval_steps_per_second: 1.769, epoch: 0.3398[0m
[32m[2022-08-31 15:05:29,114] [    INFO][0m - loss: 0.55261631, learning_rate: 2.979516497876511e-05, global_step: 2090, interval_runtime: 5.6644, interval_samples_per_second: 11.299, interval_steps_per_second: 1.765, epoch: 0.3414[0m
[32m[2022-08-31 15:05:34,776] [    INFO][0m - loss: 0.58284006, learning_rate: 2.9794184906893175e-05, global_step: 2100, interval_runtime: 5.6617, interval_samples_per_second: 11.304, interval_steps_per_second: 1.766, epoch: 0.343[0m
[32m[2022-08-31 15:05:34,776] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 15:05:34,776] [    INFO][0m -   Num examples = 12241[0m
[32m[2022-08-31 15:05:34,777] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:05:34,777] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:05:34,777] [    INFO][0m -   Total prediction steps = 192[0m
[32m[2022-08-31 15:06:29,922] [    INFO][0m - eval_loss: 0.5341325998306274, eval_accuracy: 0.7887427497753452, eval_runtime: 55.1452, eval_samples_per_second: 221.977, eval_steps_per_second: 3.482, epoch: 0.343[0m
[32m[2022-08-31 15:06:29,923] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-2100[0m
[32m[2022-08-31 15:06:29,923] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:06:31,528] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-2100/tokenizer_config.json[0m
[32m[2022-08-31 15:06:31,528] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-2100/special_tokens_map.json[0m
[32m[2022-08-31 15:06:39,615] [    INFO][0m - loss: 0.56062832, learning_rate: 2.9793204835021236e-05, global_step: 2110, interval_runtime: 64.8398, interval_samples_per_second: 0.987, interval_steps_per_second: 0.154, epoch: 0.3447[0m
[32m[2022-08-31 15:06:45,256] [    INFO][0m - loss: 0.53918185, learning_rate: 2.97922247631493e-05, global_step: 2120, interval_runtime: 5.6407, interval_samples_per_second: 11.346, interval_steps_per_second: 1.773, epoch: 0.3463[0m
[32m[2022-08-31 15:06:50,878] [    INFO][0m - loss: 0.58232565, learning_rate: 2.9791244691277362e-05, global_step: 2130, interval_runtime: 5.6213, interval_samples_per_second: 11.385, interval_steps_per_second: 1.779, epoch: 0.3479[0m
[32m[2022-08-31 15:06:56,518] [    INFO][0m - loss: 0.56039028, learning_rate: 2.9790264619405426e-05, global_step: 2140, interval_runtime: 5.6408, interval_samples_per_second: 11.346, interval_steps_per_second: 1.773, epoch: 0.3496[0m
[32m[2022-08-31 15:07:02,181] [    INFO][0m - loss: 0.60361495, learning_rate: 2.9789284547533487e-05, global_step: 2150, interval_runtime: 5.6625, interval_samples_per_second: 11.303, interval_steps_per_second: 1.766, epoch: 0.3512[0m
[32m[2022-08-31 15:07:07,826] [    INFO][0m - loss: 0.55681601, learning_rate: 2.978830447566155e-05, global_step: 2160, interval_runtime: 5.6448, interval_samples_per_second: 11.338, interval_steps_per_second: 1.772, epoch: 0.3528[0m
[32m[2022-08-31 15:07:13,489] [    INFO][0m - loss: 0.56074901, learning_rate: 2.9787324403789613e-05, global_step: 2170, interval_runtime: 5.6631, interval_samples_per_second: 11.301, interval_steps_per_second: 1.766, epoch: 0.3545[0m
[32m[2022-08-31 15:07:19,156] [    INFO][0m - loss: 0.50122747, learning_rate: 2.9786344331917674e-05, global_step: 2180, interval_runtime: 5.6668, interval_samples_per_second: 11.294, interval_steps_per_second: 1.765, epoch: 0.3561[0m
[32m[2022-08-31 15:07:24,824] [    INFO][0m - loss: 0.56284022, learning_rate: 2.9785364260045738e-05, global_step: 2190, interval_runtime: 5.6688, interval_samples_per_second: 11.29, interval_steps_per_second: 1.764, epoch: 0.3577[0m
[32m[2022-08-31 15:07:30,509] [    INFO][0m - loss: 0.51201382, learning_rate: 2.97843841881738e-05, global_step: 2200, interval_runtime: 5.6849, interval_samples_per_second: 11.258, interval_steps_per_second: 1.759, epoch: 0.3594[0m
[32m[2022-08-31 15:07:30,510] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 15:07:30,510] [    INFO][0m -   Num examples = 12241[0m
[32m[2022-08-31 15:07:30,510] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:07:30,510] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:07:30,510] [    INFO][0m -   Total prediction steps = 192[0m
[32m[2022-08-31 15:08:25,463] [    INFO][0m - eval_loss: 0.5414647459983826, eval_accuracy: 0.7857201209051549, eval_runtime: 54.9523, eval_samples_per_second: 222.757, eval_steps_per_second: 3.494, epoch: 0.3594[0m
[32m[2022-08-31 15:08:25,463] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-2200[0m
[32m[2022-08-31 15:08:25,463] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:08:27,087] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-2200/tokenizer_config.json[0m
[32m[2022-08-31 15:08:27,088] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-2200/special_tokens_map.json[0m
[32m[2022-08-31 15:08:29,496] [    INFO][0m - 
Training completed. 
[0m
[32m[2022-08-31 15:08:29,496] [    INFO][0m - Loading best model from ./checkpoints/checkpoint-1800 (score: 0.7938893881218855).[0m
[32m[2022-08-31 15:08:30,389] [    INFO][0m - train_runtime: 2567.5215, train_samples_per_second: 7629.595, train_steps_per_second: 119.22, train_loss: 0.6489110066673972, epoch: 0.3594[0m
[32m[2022-08-31 15:08:30,432] [    INFO][0m - Saving model checkpoint to ./checkpoints/[0m
[32m[2022-08-31 15:08:30,433] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 15:08:31,997] [    INFO][0m - tokenizer config file saved in ./checkpoints/tokenizer_config.json[0m
[32m[2022-08-31 15:08:31,998] [    INFO][0m - Special tokens file saved in ./checkpoints/special_tokens_map.json[0m
[32m[2022-08-31 15:08:31,999] [    INFO][0m - ***** train metrics *****[0m
[32m[2022-08-31 15:08:31,999] [    INFO][0m -   epoch                    =     0.3594[0m
[32m[2022-08-31 15:08:31,999] [    INFO][0m -   train_loss               =     0.6489[0m
[32m[2022-08-31 15:08:31,999] [    INFO][0m -   train_runtime            = 0:42:47.52[0m
[32m[2022-08-31 15:08:31,999] [    INFO][0m -   train_samples_per_second =   7629.595[0m
[32m[2022-08-31 15:08:31,999] [    INFO][0m -   train_steps_per_second   =     119.22[0m
[32m[2022-08-31 15:08:32,009] [    INFO][0m - ***** Running Prediction *****[0m
[32m[2022-08-31 15:08:32,009] [    INFO][0m -   Num examples = 13880[0m
[32m[2022-08-31 15:08:32,009] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:08:32,009] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:08:32,009] [    INFO][0m -   Total prediction steps = 217[0m
[32m[2022-08-31 15:09:34,190] [    INFO][0m - ***** test metrics *****[0m
[32m[2022-08-31 15:09:34,191] [    INFO][0m -   test_runtime            = 0:01:02.18[0m
[32m[2022-08-31 15:09:34,191] [    INFO][0m -   test_samples_per_second =    223.218[0m
[32m[2022-08-31 15:09:34,191] [    INFO][0m -   test_steps_per_second   =       3.49[0m
[32m[2022-08-31 15:09:34,191] [    INFO][0m - ***** Running Prediction *****[0m
[32m[2022-08-31 15:09:34,192] [    INFO][0m -   Num examples = 13880[0m
[32m[2022-08-31 15:09:34,192] [    INFO][0m -   Pre device batch size = 64[0m
[32m[2022-08-31 15:09:34,192] [    INFO][0m -   Total Batch size = 64[0m
[32m[2022-08-31 15:09:34,192] [    INFO][0m -   Total prediction steps = 217[0m
[32m[2022-08-31 15:10:52,052] [    INFO][0m - Predictions for cmnlif saved to ./fewclue_submit_examples.[0m
[]
run.sh: line 70: --freeze_plm: command not found
