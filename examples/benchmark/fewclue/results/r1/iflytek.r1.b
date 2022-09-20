[33m[2022-08-31 16:19:10,499] [ WARNING][0m - evaluation_strategy reset to IntervalStrategy.STEPS for do_eval is True. you can also set evaluation_strategy='epoch'.[0m
[32m[2022-08-31 16:19:10,500] [    INFO][0m - The default value for the training argument `--report_to` will change in v5 (from all installed integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as now. You should start updating your code and make this info disappear :-).[0m
[32m[2022-08-31 16:19:10,500] [    INFO][0m - ============================================================[0m
[32m[2022-08-31 16:19:10,500] [    INFO][0m -      Model Configuration Arguments      [0m
[32m[2022-08-31 16:19:10,500] [    INFO][0m - paddle commit id              :65f388690048d0965ec7d3b43fb6ee9d8c6dee7c[0m
[32m[2022-08-31 16:19:10,500] [    INFO][0m - do_save                       :True[0m
[32m[2022-08-31 16:19:10,500] [    INFO][0m - do_test                       :True[0m
[32m[2022-08-31 16:19:10,500] [    INFO][0m - early_stop_patience           :4[0m
[32m[2022-08-31 16:19:10,500] [    INFO][0m - export_type                   :paddle[0m
[32m[2022-08-31 16:19:10,500] [    INFO][0m - model_name_or_path            :roformer_v2_chinese_char_base[0m
[32m[2022-08-31 16:19:10,500] [    INFO][0m - [0m
[32m[2022-08-31 16:19:10,501] [    INFO][0m - ============================================================[0m
[32m[2022-08-31 16:19:10,501] [    INFO][0m -       Data Configuration Arguments      [0m
[32m[2022-08-31 16:19:10,501] [    INFO][0m - paddle commit id              :65f388690048d0965ec7d3b43fb6ee9d8c6dee7c[0m
[32m[2022-08-31 16:19:10,501] [    INFO][0m - encoder_hidden_size           :200[0m
[32m[2022-08-31 16:19:10,501] [    INFO][0m - prompt                        :“{'text':'text_a'}”这句话的主题是{'mask'}{'mask'}。[0m
[32m[2022-08-31 16:19:10,501] [    INFO][0m - soft_encoder                  :lstm[0m
[32m[2022-08-31 16:19:10,501] [    INFO][0m - split_id                      :few_all[0m
[32m[2022-08-31 16:19:10,501] [    INFO][0m - task_name                     :iflytek[0m
[32m[2022-08-31 16:19:10,501] [    INFO][0m - [0m
[32m[2022-08-31 16:19:10,502] [    INFO][0m - Already cached /ssd2/wanghuijuan03/.paddlenlp/models/roformer_v2_chinese_char_base/model_state.pdparams[0m
W0831 16:19:10.503507  8495 gpu_resources.cc:61] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 11.2, Runtime API Version: 11.2
W0831 16:19:10.507570  8495 gpu_resources.cc:91] device: 0, cuDNN Version: 8.1.
[32m[2022-08-31 16:19:13,506] [    INFO][0m - Already cached /ssd2/wanghuijuan03/.paddlenlp/models/roformer_v2_chinese_char_base/vocab.txt[0m
[32m[2022-08-31 16:19:13,514] [    INFO][0m - tokenizer config file saved in /ssd2/wanghuijuan03/.paddlenlp/models/roformer_v2_chinese_char_base/tokenizer_config.json[0m
[32m[2022-08-31 16:19:13,515] [    INFO][0m - Special tokens file saved in /ssd2/wanghuijuan03/.paddlenlp/models/roformer_v2_chinese_char_base/special_tokens_map.json[0m
[32m[2022-08-31 16:19:13,515] [    INFO][0m - Using template: [{'add_prefix_space': '', 'hard': '“'}, {'add_prefix_space': '', 'text': 'text_a'}, {'add_prefix_space': '', 'hard': '”这句话的主题是'}, {'add_prefix_space': '', 'mask': None}, {'add_prefix_space': '', 'mask': None}, {'add_prefix_space': '', 'hard': '。'}][0m
2022-08-31 16:19:13,540 INFO [download.py:119] unique_endpoints {''}
[32m[2022-08-31 16:19:13,738] [    INFO][0m - ============================================================[0m
[32m[2022-08-31 16:19:13,738] [    INFO][0m -     Training Configuration Arguments    [0m
[32m[2022-08-31 16:19:13,738] [    INFO][0m - paddle commit id              :65f388690048d0965ec7d3b43fb6ee9d8c6dee7c[0m
[32m[2022-08-31 16:19:13,738] [    INFO][0m - _no_sync_in_gradient_accumulation:True[0m
[32m[2022-08-31 16:19:13,738] [    INFO][0m - adam_beta1                    :0.9[0m
[32m[2022-08-31 16:19:13,738] [    INFO][0m - adam_beta2                    :0.999[0m
[32m[2022-08-31 16:19:13,738] [    INFO][0m - adam_epsilon                  :1e-08[0m
[32m[2022-08-31 16:19:13,739] [    INFO][0m - alpha_rdrop                   :5.0[0m
[32m[2022-08-31 16:19:13,739] [    INFO][0m - alpha_rgl                     :0.5[0m
[32m[2022-08-31 16:19:13,739] [    INFO][0m - current_device                :gpu:0[0m
[32m[2022-08-31 16:19:13,739] [    INFO][0m - dataloader_drop_last          :False[0m
[32m[2022-08-31 16:19:13,739] [    INFO][0m - dataloader_num_workers        :0[0m
[32m[2022-08-31 16:19:13,739] [    INFO][0m - device                        :gpu[0m
[32m[2022-08-31 16:19:13,739] [    INFO][0m - disable_tqdm                  :True[0m
[32m[2022-08-31 16:19:13,739] [    INFO][0m - do_eval                       :True[0m
[32m[2022-08-31 16:19:13,739] [    INFO][0m - do_export                     :False[0m
[32m[2022-08-31 16:19:13,739] [    INFO][0m - do_predict                    :True[0m
[32m[2022-08-31 16:19:13,739] [    INFO][0m - do_train                      :True[0m
[32m[2022-08-31 16:19:13,739] [    INFO][0m - eval_batch_size               :32[0m
[32m[2022-08-31 16:19:13,740] [    INFO][0m - eval_steps                    :100[0m
[32m[2022-08-31 16:19:13,740] [    INFO][0m - evaluation_strategy           :IntervalStrategy.STEPS[0m
[32m[2022-08-31 16:19:13,740] [    INFO][0m - first_max_length              :None[0m
[32m[2022-08-31 16:19:13,740] [    INFO][0m - fp16                          :False[0m
[32m[2022-08-31 16:19:13,740] [    INFO][0m - fp16_opt_level                :O1[0m
[32m[2022-08-31 16:19:13,740] [    INFO][0m - freeze_dropout                :False[0m
[32m[2022-08-31 16:19:13,740] [    INFO][0m - freeze_plm                    :False[0m
[32m[2022-08-31 16:19:13,740] [    INFO][0m - gradient_accumulation_steps   :1[0m
[32m[2022-08-31 16:19:13,740] [    INFO][0m - greater_is_better             :True[0m
[32m[2022-08-31 16:19:13,740] [    INFO][0m - ignore_data_skip              :False[0m
[32m[2022-08-31 16:19:13,740] [    INFO][0m - label_names                   :None[0m
[32m[2022-08-31 16:19:13,740] [    INFO][0m - learning_rate                 :3e-05[0m
[32m[2022-08-31 16:19:13,740] [    INFO][0m - load_best_model_at_end        :True[0m
[32m[2022-08-31 16:19:13,741] [    INFO][0m - local_process_index           :0[0m
[32m[2022-08-31 16:19:13,741] [    INFO][0m - local_rank                    :-1[0m
[32m[2022-08-31 16:19:13,741] [    INFO][0m - log_level                     :-1[0m
[32m[2022-08-31 16:19:13,741] [    INFO][0m - log_level_replica             :-1[0m
[32m[2022-08-31 16:19:13,741] [    INFO][0m - log_on_each_node              :True[0m
[32m[2022-08-31 16:19:13,741] [    INFO][0m - logging_dir                   :./checkpoints/runs/Aug31_16-19-10_instance-3bwob41y-01[0m
[32m[2022-08-31 16:19:13,741] [    INFO][0m - logging_first_step            :False[0m
[32m[2022-08-31 16:19:13,741] [    INFO][0m - logging_steps                 :10[0m
[32m[2022-08-31 16:19:13,741] [    INFO][0m - logging_strategy              :IntervalStrategy.STEPS[0m
[32m[2022-08-31 16:19:13,741] [    INFO][0m - lr_scheduler_type             :SchedulerType.LINEAR[0m
[32m[2022-08-31 16:19:13,741] [    INFO][0m - max_grad_norm                 :1.0[0m
[32m[2022-08-31 16:19:13,741] [    INFO][0m - max_seq_length                :320[0m
[32m[2022-08-31 16:19:13,741] [    INFO][0m - max_steps                     :-1[0m
[32m[2022-08-31 16:19:13,742] [    INFO][0m - metric_for_best_model         :accuracy[0m
[32m[2022-08-31 16:19:13,742] [    INFO][0m - minimum_eval_times            :None[0m
[32m[2022-08-31 16:19:13,742] [    INFO][0m - no_cuda                       :False[0m
[32m[2022-08-31 16:19:13,742] [    INFO][0m - num_train_epochs              :100.0[0m
[32m[2022-08-31 16:19:13,742] [    INFO][0m - optim                         :OptimizerNames.ADAMW[0m
[32m[2022-08-31 16:19:13,742] [    INFO][0m - other_max_length              :None[0m
[32m[2022-08-31 16:19:13,742] [    INFO][0m - output_dir                    :./checkpoints/[0m
[32m[2022-08-31 16:19:13,742] [    INFO][0m - overwrite_output_dir          :False[0m
[32m[2022-08-31 16:19:13,742] [    INFO][0m - past_index                    :-1[0m
[32m[2022-08-31 16:19:13,742] [    INFO][0m - per_device_eval_batch_size    :32[0m
[32m[2022-08-31 16:19:13,742] [    INFO][0m - per_device_train_batch_size   :8[0m
[32m[2022-08-31 16:19:13,742] [    INFO][0m - ppt_adam_beta1                :0.9[0m
[32m[2022-08-31 16:19:13,742] [    INFO][0m - ppt_adam_beta2                :0.999[0m
[32m[2022-08-31 16:19:13,743] [    INFO][0m - ppt_adam_epsilon              :1e-08[0m
[32m[2022-08-31 16:19:13,743] [    INFO][0m - ppt_learning_rate             :0.0003[0m
[32m[2022-08-31 16:19:13,743] [    INFO][0m - ppt_weight_decay              :0.0[0m
[32m[2022-08-31 16:19:13,743] [    INFO][0m - prediction_loss_only          :False[0m
[32m[2022-08-31 16:19:13,743] [    INFO][0m - process_index                 :0[0m
[32m[2022-08-31 16:19:13,743] [    INFO][0m - remove_unused_columns         :True[0m
[32m[2022-08-31 16:19:13,743] [    INFO][0m - report_to                     :['visualdl'][0m
[32m[2022-08-31 16:19:13,743] [    INFO][0m - resume_from_checkpoint        :None[0m
[32m[2022-08-31 16:19:13,743] [    INFO][0m - run_name                      :./checkpoints/[0m
[32m[2022-08-31 16:19:13,743] [    INFO][0m - save_on_each_node             :False[0m
[32m[2022-08-31 16:19:13,743] [    INFO][0m - save_steps                    :100[0m
[32m[2022-08-31 16:19:13,743] [    INFO][0m - save_strategy                 :IntervalStrategy.STEPS[0m
[32m[2022-08-31 16:19:13,743] [    INFO][0m - save_total_limit              :None[0m
[32m[2022-08-31 16:19:13,743] [    INFO][0m - scale_loss                    :32768[0m
[32m[2022-08-31 16:19:13,744] [    INFO][0m - seed                          :42[0m
[32m[2022-08-31 16:19:13,744] [    INFO][0m - should_log                    :True[0m
[32m[2022-08-31 16:19:13,744] [    INFO][0m - should_save                   :True[0m
[32m[2022-08-31 16:19:13,744] [    INFO][0m - task_type                     :multi-class[0m
[32m[2022-08-31 16:19:13,744] [    INFO][0m - train_batch_size              :8[0m
[32m[2022-08-31 16:19:13,744] [    INFO][0m - truncate_mode                 :tail[0m
[32m[2022-08-31 16:19:13,744] [    INFO][0m - use_rdrop                     :False[0m
[32m[2022-08-31 16:19:13,744] [    INFO][0m - use_rgl                       :False[0m
[32m[2022-08-31 16:19:13,744] [    INFO][0m - warmup_ratio                  :0.0[0m
[32m[2022-08-31 16:19:13,744] [    INFO][0m - warmup_steps                  :0[0m
[32m[2022-08-31 16:19:13,744] [    INFO][0m - weight_decay                  :0.0[0m
[32m[2022-08-31 16:19:13,744] [    INFO][0m - world_size                    :1[0m
[32m[2022-08-31 16:19:13,744] [    INFO][0m - [0m
[32m[2022-08-31 16:19:13,747] [    INFO][0m - ***** Running training *****[0m
[32m[2022-08-31 16:19:13,747] [    INFO][0m -   Num examples = 3024[0m
[32m[2022-08-31 16:19:13,747] [    INFO][0m -   Num Epochs = 100[0m
[32m[2022-08-31 16:19:13,747] [    INFO][0m -   Instantaneous batch size per device = 8[0m
[32m[2022-08-31 16:19:13,747] [    INFO][0m -   Total train batch size (w. parallel, distributed & accumulation) = 8[0m
[32m[2022-08-31 16:19:13,747] [    INFO][0m -   Gradient Accumulation steps = 1[0m
[32m[2022-08-31 16:19:13,748] [    INFO][0m -   Total optimization steps = 37800.0[0m
[32m[2022-08-31 16:19:13,748] [    INFO][0m -   Total num train samples = 302400[0m
[32m[2022-08-31 16:19:16,008] [    INFO][0m - loss: 3.25806541, learning_rate: 2.999206349206349e-05, global_step: 10, interval_runtime: 2.2591, interval_samples_per_second: 3.541, interval_steps_per_second: 4.427, epoch: 0.0265[0m
[32m[2022-08-31 16:19:17,556] [    INFO][0m - loss: 2.78513584, learning_rate: 2.9984126984126986e-05, global_step: 20, interval_runtime: 1.5474, interval_samples_per_second: 5.17, interval_steps_per_second: 6.463, epoch: 0.0529[0m
[32m[2022-08-31 16:19:19,105] [    INFO][0m - loss: 2.88013115, learning_rate: 2.9976190476190477e-05, global_step: 30, interval_runtime: 1.5498, interval_samples_per_second: 5.162, interval_steps_per_second: 6.452, epoch: 0.0794[0m
[32m[2022-08-31 16:19:20,652] [    INFO][0m - loss: 2.47297802, learning_rate: 2.9968253968253967e-05, global_step: 40, interval_runtime: 1.5467, interval_samples_per_second: 5.172, interval_steps_per_second: 6.465, epoch: 0.1058[0m
[32m[2022-08-31 16:19:22,209] [    INFO][0m - loss: 2.82525558, learning_rate: 2.996031746031746e-05, global_step: 50, interval_runtime: 1.5571, interval_samples_per_second: 5.138, interval_steps_per_second: 6.422, epoch: 0.1323[0m
[32m[2022-08-31 16:19:23,770] [    INFO][0m - loss: 2.93165932, learning_rate: 2.9952380952380952e-05, global_step: 60, interval_runtime: 1.5612, interval_samples_per_second: 5.124, interval_steps_per_second: 6.405, epoch: 0.1587[0m
[32m[2022-08-31 16:19:25,316] [    INFO][0m - loss: 2.84620743, learning_rate: 2.9944444444444443e-05, global_step: 70, interval_runtime: 1.5451, interval_samples_per_second: 5.178, interval_steps_per_second: 6.472, epoch: 0.1852[0m
[32m[2022-08-31 16:19:26,854] [    INFO][0m - loss: 2.06664562, learning_rate: 2.9936507936507937e-05, global_step: 80, interval_runtime: 1.5393, interval_samples_per_second: 5.197, interval_steps_per_second: 6.496, epoch: 0.2116[0m
[32m[2022-08-31 16:19:28,405] [    INFO][0m - loss: 2.29035492, learning_rate: 2.9928571428571428e-05, global_step: 90, interval_runtime: 1.5503, interval_samples_per_second: 5.16, interval_steps_per_second: 6.45, epoch: 0.2381[0m
[32m[2022-08-31 16:19:29,953] [    INFO][0m - loss: 2.35693874, learning_rate: 2.992063492063492e-05, global_step: 100, interval_runtime: 1.5483, interval_samples_per_second: 5.167, interval_steps_per_second: 6.459, epoch: 0.2646[0m
[32m[2022-08-31 16:19:29,954] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 16:19:29,954] [    INFO][0m -   Num examples = 1373[0m
[32m[2022-08-31 16:19:29,954] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 16:19:29,954] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 16:19:29,954] [    INFO][0m -   Total prediction steps = 43[0m
[32m[2022-08-31 16:19:42,031] [    INFO][0m - eval_loss: 2.104221820831299, eval_accuracy: 0.4996358339402768, eval_runtime: 12.0756, eval_samples_per_second: 113.701, eval_steps_per_second: 3.561, epoch: 0.2646[0m
[32m[2022-08-31 16:19:42,031] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-100[0m
[32m[2022-08-31 16:19:42,032] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 16:19:43,909] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-100/tokenizer_config.json[0m
[32m[2022-08-31 16:19:43,910] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-100/special_tokens_map.json[0m
[32m[2022-08-31 16:19:49,642] [    INFO][0m - loss: 2.04689674, learning_rate: 2.9912698412698416e-05, global_step: 110, interval_runtime: 19.6886, interval_samples_per_second: 0.406, interval_steps_per_second: 0.508, epoch: 0.291[0m
[32m[2022-08-31 16:19:51,186] [    INFO][0m - loss: 2.29069748, learning_rate: 2.9904761904761907e-05, global_step: 120, interval_runtime: 1.5441, interval_samples_per_second: 5.181, interval_steps_per_second: 6.476, epoch: 0.3175[0m
[32m[2022-08-31 16:19:52,724] [    INFO][0m - loss: 2.0772438, learning_rate: 2.9896825396825398e-05, global_step: 130, interval_runtime: 1.5383, interval_samples_per_second: 5.201, interval_steps_per_second: 6.501, epoch: 0.3439[0m
[32m[2022-08-31 16:19:54,272] [    INFO][0m - loss: 2.14852028, learning_rate: 2.9888888888888892e-05, global_step: 140, interval_runtime: 1.5481, interval_samples_per_second: 5.168, interval_steps_per_second: 6.459, epoch: 0.3704[0m
[32m[2022-08-31 16:19:55,821] [    INFO][0m - loss: 2.3258358, learning_rate: 2.9880952380952383e-05, global_step: 150, interval_runtime: 1.5492, interval_samples_per_second: 5.164, interval_steps_per_second: 6.455, epoch: 0.3968[0m
[32m[2022-08-31 16:19:57,386] [    INFO][0m - loss: 2.60412121, learning_rate: 2.9873015873015874e-05, global_step: 160, interval_runtime: 1.5637, interval_samples_per_second: 5.116, interval_steps_per_second: 6.395, epoch: 0.4233[0m
[32m[2022-08-31 16:19:58,937] [    INFO][0m - loss: 1.90838509, learning_rate: 2.9865079365079368e-05, global_step: 170, interval_runtime: 1.5518, interval_samples_per_second: 5.155, interval_steps_per_second: 6.444, epoch: 0.4497[0m
[32m[2022-08-31 16:20:00,491] [    INFO][0m - loss: 2.32103481, learning_rate: 2.985714285714286e-05, global_step: 180, interval_runtime: 1.5537, interval_samples_per_second: 5.149, interval_steps_per_second: 6.436, epoch: 0.4762[0m
[32m[2022-08-31 16:20:02,051] [    INFO][0m - loss: 1.85383759, learning_rate: 2.984920634920635e-05, global_step: 190, interval_runtime: 1.5608, interval_samples_per_second: 5.126, interval_steps_per_second: 6.407, epoch: 0.5026[0m
[32m[2022-08-31 16:20:03,613] [    INFO][0m - loss: 2.09193916, learning_rate: 2.984126984126984e-05, global_step: 200, interval_runtime: 1.561, interval_samples_per_second: 5.125, interval_steps_per_second: 6.406, epoch: 0.5291[0m
[32m[2022-08-31 16:20:03,614] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 16:20:03,614] [    INFO][0m -   Num examples = 1373[0m
[32m[2022-08-31 16:20:03,614] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 16:20:03,614] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 16:20:03,614] [    INFO][0m -   Total prediction steps = 43[0m
[32m[2022-08-31 16:20:15,578] [    INFO][0m - eval_loss: 1.9556325674057007, eval_accuracy: 0.5163874726875455, eval_runtime: 11.9632, eval_samples_per_second: 114.769, eval_steps_per_second: 3.594, epoch: 0.5291[0m
[32m[2022-08-31 16:20:15,578] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-200[0m
[32m[2022-08-31 16:20:15,579] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 16:20:17,345] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-200/tokenizer_config.json[0m
[32m[2022-08-31 16:20:17,345] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-200/special_tokens_map.json[0m
[32m[2022-08-31 16:20:23,336] [    INFO][0m - loss: 1.87235718, learning_rate: 2.9833333333333335e-05, global_step: 210, interval_runtime: 19.7234, interval_samples_per_second: 0.406, interval_steps_per_second: 0.507, epoch: 0.5556[0m
[32m[2022-08-31 16:20:24,889] [    INFO][0m - loss: 2.16832848, learning_rate: 2.9825396825396825e-05, global_step: 220, interval_runtime: 1.5532, interval_samples_per_second: 5.151, interval_steps_per_second: 6.438, epoch: 0.582[0m
[32m[2022-08-31 16:20:26,432] [    INFO][0m - loss: 1.94843979, learning_rate: 2.9817460317460316e-05, global_step: 230, interval_runtime: 1.5435, interval_samples_per_second: 5.183, interval_steps_per_second: 6.479, epoch: 0.6085[0m
[32m[2022-08-31 16:20:27,968] [    INFO][0m - loss: 1.79297085, learning_rate: 2.980952380952381e-05, global_step: 240, interval_runtime: 1.5351, interval_samples_per_second: 5.211, interval_steps_per_second: 6.514, epoch: 0.6349[0m
[32m[2022-08-31 16:20:29,518] [    INFO][0m - loss: 1.76371746, learning_rate: 2.98015873015873e-05, global_step: 250, interval_runtime: 1.5497, interval_samples_per_second: 5.162, interval_steps_per_second: 6.453, epoch: 0.6614[0m
[32m[2022-08-31 16:20:31,069] [    INFO][0m - loss: 1.85072174, learning_rate: 2.9793650793650792e-05, global_step: 260, interval_runtime: 1.5514, interval_samples_per_second: 5.157, interval_steps_per_second: 6.446, epoch: 0.6878[0m
[32m[2022-08-31 16:20:32,617] [    INFO][0m - loss: 1.67825661, learning_rate: 2.9785714285714286e-05, global_step: 270, interval_runtime: 1.5485, interval_samples_per_second: 5.166, interval_steps_per_second: 6.458, epoch: 0.7143[0m
[32m[2022-08-31 16:20:34,163] [    INFO][0m - loss: 2.12542496, learning_rate: 2.9777777777777777e-05, global_step: 280, interval_runtime: 1.5457, interval_samples_per_second: 5.176, interval_steps_per_second: 6.469, epoch: 0.7407[0m
[32m[2022-08-31 16:20:35,713] [    INFO][0m - loss: 1.86569176, learning_rate: 2.9769841269841268e-05, global_step: 290, interval_runtime: 1.5504, interval_samples_per_second: 5.16, interval_steps_per_second: 6.45, epoch: 0.7672[0m
[32m[2022-08-31 16:20:37,250] [    INFO][0m - loss: 1.76646957, learning_rate: 2.9761904761904762e-05, global_step: 300, interval_runtime: 1.5364, interval_samples_per_second: 5.207, interval_steps_per_second: 6.509, epoch: 0.7937[0m
[32m[2022-08-31 16:20:37,251] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 16:20:37,251] [    INFO][0m -   Num examples = 1373[0m
[32m[2022-08-31 16:20:37,251] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 16:20:37,251] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 16:20:37,251] [    INFO][0m -   Total prediction steps = 43[0m
[32m[2022-08-31 16:20:49,167] [    INFO][0m - eval_loss: 1.8884632587432861, eval_accuracy: 0.5316824471959214, eval_runtime: 11.9158, eval_samples_per_second: 115.225, eval_steps_per_second: 3.609, epoch: 0.7937[0m
[32m[2022-08-31 16:20:49,168] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-300[0m
[32m[2022-08-31 16:20:49,168] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 16:20:50,973] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-300/tokenizer_config.json[0m
[32m[2022-08-31 16:20:50,973] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-300/special_tokens_map.json[0m
[32m[2022-08-31 16:20:56,692] [    INFO][0m - loss: 1.97669239, learning_rate: 2.9753968253968256e-05, global_step: 310, interval_runtime: 19.4427, interval_samples_per_second: 0.411, interval_steps_per_second: 0.514, epoch: 0.8201[0m
[32m[2022-08-31 16:20:58,235] [    INFO][0m - loss: 1.6027977, learning_rate: 2.9746031746031747e-05, global_step: 320, interval_runtime: 1.5426, interval_samples_per_second: 5.186, interval_steps_per_second: 6.482, epoch: 0.8466[0m
[32m[2022-08-31 16:20:59,784] [    INFO][0m - loss: 1.89426155, learning_rate: 2.973809523809524e-05, global_step: 330, interval_runtime: 1.5489, interval_samples_per_second: 5.165, interval_steps_per_second: 6.456, epoch: 0.873[0m
[32m[2022-08-31 16:21:01,330] [    INFO][0m - loss: 1.79776821, learning_rate: 2.9730158730158732e-05, global_step: 340, interval_runtime: 1.5463, interval_samples_per_second: 5.174, interval_steps_per_second: 6.467, epoch: 0.8995[0m
[32m[2022-08-31 16:21:02,875] [    INFO][0m - loss: 2.11113396, learning_rate: 2.9722222222222223e-05, global_step: 350, interval_runtime: 1.5448, interval_samples_per_second: 5.179, interval_steps_per_second: 6.473, epoch: 0.9259[0m
[32m[2022-08-31 16:21:04,412] [    INFO][0m - loss: 2.29682522, learning_rate: 2.9714285714285717e-05, global_step: 360, interval_runtime: 1.5373, interval_samples_per_second: 5.204, interval_steps_per_second: 6.505, epoch: 0.9524[0m
[32m[2022-08-31 16:21:05,955] [    INFO][0m - loss: 1.8502964, learning_rate: 2.9706349206349208e-05, global_step: 370, interval_runtime: 1.5421, interval_samples_per_second: 5.188, interval_steps_per_second: 6.485, epoch: 0.9788[0m
[32m[2022-08-31 16:21:07,634] [    INFO][0m - loss: 1.42427845, learning_rate: 2.96984126984127e-05, global_step: 380, interval_runtime: 1.6795, interval_samples_per_second: 4.763, interval_steps_per_second: 5.954, epoch: 1.0053[0m
[32m[2022-08-31 16:21:09,179] [    INFO][0m - loss: 1.34523573, learning_rate: 2.9690476190476193e-05, global_step: 390, interval_runtime: 1.5448, interval_samples_per_second: 5.179, interval_steps_per_second: 6.473, epoch: 1.0317[0m
[32m[2022-08-31 16:21:10,726] [    INFO][0m - loss: 0.97555046, learning_rate: 2.9682539682539683e-05, global_step: 400, interval_runtime: 1.5468, interval_samples_per_second: 5.172, interval_steps_per_second: 6.465, epoch: 1.0582[0m
[32m[2022-08-31 16:21:10,726] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 16:21:10,727] [    INFO][0m -   Num examples = 1373[0m
[32m[2022-08-31 16:21:10,727] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 16:21:10,727] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 16:21:10,727] [    INFO][0m -   Total prediction steps = 43[0m
[32m[2022-08-31 16:21:22,676] [    INFO][0m - eval_loss: 1.981291651725769, eval_accuracy: 0.5345957756737072, eval_runtime: 11.9483, eval_samples_per_second: 114.912, eval_steps_per_second: 3.599, epoch: 1.0582[0m
[32m[2022-08-31 16:21:22,676] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-400[0m
[32m[2022-08-31 16:21:22,676] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 16:21:24,563] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-400/tokenizer_config.json[0m
[32m[2022-08-31 16:21:24,563] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-400/special_tokens_map.json[0m
[32m[2022-08-31 16:21:31,605] [    INFO][0m - loss: 1.10820179, learning_rate: 2.9674603174603174e-05, global_step: 410, interval_runtime: 20.8793, interval_samples_per_second: 0.383, interval_steps_per_second: 0.479, epoch: 1.0847[0m
[32m[2022-08-31 16:21:33,148] [    INFO][0m - loss: 1.2164712, learning_rate: 2.966666666666667e-05, global_step: 420, interval_runtime: 1.5433, interval_samples_per_second: 5.184, interval_steps_per_second: 6.48, epoch: 1.1111[0m
[32m[2022-08-31 16:21:34,695] [    INFO][0m - loss: 1.07264252, learning_rate: 2.965873015873016e-05, global_step: 430, interval_runtime: 1.5471, interval_samples_per_second: 5.171, interval_steps_per_second: 6.464, epoch: 1.1376[0m
[32m[2022-08-31 16:21:36,242] [    INFO][0m - loss: 1.14051447, learning_rate: 2.965079365079365e-05, global_step: 440, interval_runtime: 1.5466, interval_samples_per_second: 5.173, interval_steps_per_second: 6.466, epoch: 1.164[0m
[32m[2022-08-31 16:21:37,782] [    INFO][0m - loss: 1.21917229, learning_rate: 2.9642857142857144e-05, global_step: 450, interval_runtime: 1.5401, interval_samples_per_second: 5.194, interval_steps_per_second: 6.493, epoch: 1.1905[0m
[32m[2022-08-31 16:21:39,322] [    INFO][0m - loss: 1.3367878, learning_rate: 2.9634920634920635e-05, global_step: 460, interval_runtime: 1.5392, interval_samples_per_second: 5.197, interval_steps_per_second: 6.497, epoch: 1.2169[0m
[32m[2022-08-31 16:21:40,865] [    INFO][0m - loss: 1.19555731, learning_rate: 2.9626984126984126e-05, global_step: 470, interval_runtime: 1.5437, interval_samples_per_second: 5.182, interval_steps_per_second: 6.478, epoch: 1.2434[0m
[32m[2022-08-31 16:21:42,419] [    INFO][0m - loss: 1.17201357, learning_rate: 2.961904761904762e-05, global_step: 480, interval_runtime: 1.5541, interval_samples_per_second: 5.148, interval_steps_per_second: 6.434, epoch: 1.2698[0m
[32m[2022-08-31 16:21:43,954] [    INFO][0m - loss: 1.28550529, learning_rate: 2.961111111111111e-05, global_step: 490, interval_runtime: 1.5349, interval_samples_per_second: 5.212, interval_steps_per_second: 6.515, epoch: 1.2963[0m
[32m[2022-08-31 16:21:45,506] [    INFO][0m - loss: 1.58395691, learning_rate: 2.96031746031746e-05, global_step: 500, interval_runtime: 1.5522, interval_samples_per_second: 5.154, interval_steps_per_second: 6.443, epoch: 1.3228[0m
[32m[2022-08-31 16:21:45,507] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 16:21:45,507] [    INFO][0m -   Num examples = 1373[0m
[32m[2022-08-31 16:21:45,507] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 16:21:45,507] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 16:21:45,507] [    INFO][0m -   Total prediction steps = 43[0m
[32m[2022-08-31 16:21:57,446] [    INFO][0m - eval_loss: 1.905693769454956, eval_accuracy: 0.5520757465404225, eval_runtime: 11.938, eval_samples_per_second: 115.011, eval_steps_per_second: 3.602, epoch: 1.3228[0m
[32m[2022-08-31 16:21:57,446] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-500[0m
[32m[2022-08-31 16:21:57,446] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 16:21:59,665] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-500/tokenizer_config.json[0m
[32m[2022-08-31 16:21:59,665] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-500/special_tokens_map.json[0m
[32m[2022-08-31 16:22:05,283] [    INFO][0m - loss: 1.61620026, learning_rate: 2.95952380952381e-05, global_step: 510, interval_runtime: 19.7764, interval_samples_per_second: 0.405, interval_steps_per_second: 0.506, epoch: 1.3492[0m
[32m[2022-08-31 16:22:06,820] [    INFO][0m - loss: 1.175105, learning_rate: 2.958730158730159e-05, global_step: 520, interval_runtime: 1.5368, interval_samples_per_second: 5.206, interval_steps_per_second: 6.507, epoch: 1.3757[0m
[32m[2022-08-31 16:22:08,358] [    INFO][0m - loss: 1.07922726, learning_rate: 2.957936507936508e-05, global_step: 530, interval_runtime: 1.5382, interval_samples_per_second: 5.201, interval_steps_per_second: 6.501, epoch: 1.4021[0m
[32m[2022-08-31 16:22:09,921] [    INFO][0m - loss: 1.34318628, learning_rate: 2.9571428571428575e-05, global_step: 540, interval_runtime: 1.563, interval_samples_per_second: 5.118, interval_steps_per_second: 6.398, epoch: 1.4286[0m
[32m[2022-08-31 16:22:11,468] [    INFO][0m - loss: 1.19761763, learning_rate: 2.9563492063492066e-05, global_step: 550, interval_runtime: 1.5478, interval_samples_per_second: 5.169, interval_steps_per_second: 6.461, epoch: 1.455[0m
[32m[2022-08-31 16:22:13,008] [    INFO][0m - loss: 1.34640808, learning_rate: 2.9555555555555556e-05, global_step: 560, interval_runtime: 1.5392, interval_samples_per_second: 5.197, interval_steps_per_second: 6.497, epoch: 1.4815[0m
[32m[2022-08-31 16:22:14,543] [    INFO][0m - loss: 1.42163773, learning_rate: 2.954761904761905e-05, global_step: 570, interval_runtime: 1.5354, interval_samples_per_second: 5.21, interval_steps_per_second: 6.513, epoch: 1.5079[0m
[32m[2022-08-31 16:22:16,083] [    INFO][0m - loss: 1.32896833, learning_rate: 2.953968253968254e-05, global_step: 580, interval_runtime: 1.54, interval_samples_per_second: 5.195, interval_steps_per_second: 6.494, epoch: 1.5344[0m
[32m[2022-08-31 16:22:17,626] [    INFO][0m - loss: 1.68688221, learning_rate: 2.9531746031746032e-05, global_step: 590, interval_runtime: 1.5428, interval_samples_per_second: 5.185, interval_steps_per_second: 6.482, epoch: 1.5608[0m
[32m[2022-08-31 16:22:19,173] [    INFO][0m - loss: 1.40083504, learning_rate: 2.9523809523809523e-05, global_step: 600, interval_runtime: 1.5474, interval_samples_per_second: 5.17, interval_steps_per_second: 6.462, epoch: 1.5873[0m
[32m[2022-08-31 16:22:19,174] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 16:22:19,178] [    INFO][0m -   Num examples = 1373[0m
[32m[2022-08-31 16:22:19,178] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 16:22:19,178] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 16:22:19,178] [    INFO][0m -   Total prediction steps = 43[0m
[32m[2022-08-31 16:22:31,097] [    INFO][0m - eval_loss: 1.8756550550460815, eval_accuracy: 0.5593590677348871, eval_runtime: 11.9224, eval_samples_per_second: 115.162, eval_steps_per_second: 3.607, epoch: 1.5873[0m
[32m[2022-08-31 16:22:31,098] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-600[0m
[32m[2022-08-31 16:22:31,098] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 16:22:32,961] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-600/tokenizer_config.json[0m
[32m[2022-08-31 16:22:32,961] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-600/special_tokens_map.json[0m
[32m[2022-08-31 16:22:38,427] [    INFO][0m - loss: 1.29296532, learning_rate: 2.9515873015873017e-05, global_step: 610, interval_runtime: 19.2535, interval_samples_per_second: 0.416, interval_steps_per_second: 0.519, epoch: 1.6138[0m
[32m[2022-08-31 16:22:39,968] [    INFO][0m - loss: 1.41460857, learning_rate: 2.9507936507936508e-05, global_step: 620, interval_runtime: 1.5372, interval_samples_per_second: 5.204, interval_steps_per_second: 6.505, epoch: 1.6402[0m
[32m[2022-08-31 16:22:41,509] [    INFO][0m - loss: 1.07771969, learning_rate: 2.95e-05, global_step: 630, interval_runtime: 1.5445, interval_samples_per_second: 5.18, interval_steps_per_second: 6.474, epoch: 1.6667[0m
[32m[2022-08-31 16:22:43,068] [    INFO][0m - loss: 1.3570714, learning_rate: 2.9492063492063493e-05, global_step: 640, interval_runtime: 1.5597, interval_samples_per_second: 5.129, interval_steps_per_second: 6.412, epoch: 1.6931[0m
[32m[2022-08-31 16:22:44,617] [    INFO][0m - loss: 1.38629341, learning_rate: 2.9484126984126984e-05, global_step: 650, interval_runtime: 1.5486, interval_samples_per_second: 5.166, interval_steps_per_second: 6.457, epoch: 1.7196[0m
[32m[2022-08-31 16:22:46,155] [    INFO][0m - loss: 1.64373341, learning_rate: 2.9476190476190475e-05, global_step: 660, interval_runtime: 1.5383, interval_samples_per_second: 5.2, interval_steps_per_second: 6.501, epoch: 1.746[0m
[32m[2022-08-31 16:22:47,692] [    INFO][0m - loss: 1.37485762, learning_rate: 2.946825396825397e-05, global_step: 670, interval_runtime: 1.5366, interval_samples_per_second: 5.206, interval_steps_per_second: 6.508, epoch: 1.7725[0m
[32m[2022-08-31 16:22:49,237] [    INFO][0m - loss: 1.64209957, learning_rate: 2.946031746031746e-05, global_step: 680, interval_runtime: 1.5452, interval_samples_per_second: 5.177, interval_steps_per_second: 6.472, epoch: 1.7989[0m
[32m[2022-08-31 16:22:50,793] [    INFO][0m - loss: 1.29362822, learning_rate: 2.945238095238095e-05, global_step: 690, interval_runtime: 1.5555, interval_samples_per_second: 5.143, interval_steps_per_second: 6.429, epoch: 1.8254[0m
[32m[2022-08-31 16:22:52,343] [    INFO][0m - loss: 1.27164164, learning_rate: 2.9444444444444445e-05, global_step: 700, interval_runtime: 1.5497, interval_samples_per_second: 5.162, interval_steps_per_second: 6.453, epoch: 1.8519[0m
[32m[2022-08-31 16:22:52,343] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 16:22:52,343] [    INFO][0m -   Num examples = 1373[0m
[32m[2022-08-31 16:22:52,343] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 16:22:52,344] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 16:22:52,344] [    INFO][0m -   Total prediction steps = 43[0m
[32m[2022-08-31 16:23:04,197] [    INFO][0m - eval_loss: 1.9495265483856201, eval_accuracy: 0.526584122359796, eval_runtime: 11.8533, eval_samples_per_second: 115.833, eval_steps_per_second: 3.628, epoch: 1.8519[0m
[32m[2022-08-31 16:23:04,198] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-700[0m
[32m[2022-08-31 16:23:04,198] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 16:23:06,131] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-700/tokenizer_config.json[0m
[32m[2022-08-31 16:23:06,132] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-700/special_tokens_map.json[0m
[32m[2022-08-31 16:23:11,804] [    INFO][0m - loss: 1.36545229, learning_rate: 2.943650793650794e-05, global_step: 710, interval_runtime: 19.4614, interval_samples_per_second: 0.411, interval_steps_per_second: 0.514, epoch: 1.8783[0m
[32m[2022-08-31 16:23:13,352] [    INFO][0m - loss: 1.19981756, learning_rate: 2.942857142857143e-05, global_step: 720, interval_runtime: 1.5481, interval_samples_per_second: 5.168, interval_steps_per_second: 6.46, epoch: 1.9048[0m
[32m[2022-08-31 16:23:14,901] [    INFO][0m - loss: 1.50650177, learning_rate: 2.9420634920634924e-05, global_step: 730, interval_runtime: 1.5485, interval_samples_per_second: 5.166, interval_steps_per_second: 6.458, epoch: 1.9312[0m
[32m[2022-08-31 16:23:16,445] [    INFO][0m - loss: 1.3464819, learning_rate: 2.9412698412698414e-05, global_step: 740, interval_runtime: 1.5448, interval_samples_per_second: 5.179, interval_steps_per_second: 6.473, epoch: 1.9577[0m
[32m[2022-08-31 16:23:17,998] [    INFO][0m - loss: 1.08746042, learning_rate: 2.9404761904761905e-05, global_step: 750, interval_runtime: 1.553, interval_samples_per_second: 5.151, interval_steps_per_second: 6.439, epoch: 1.9841[0m
[32m[2022-08-31 16:23:19,650] [    INFO][0m - loss: 0.95648546, learning_rate: 2.93968253968254e-05, global_step: 760, interval_runtime: 1.6519, interval_samples_per_second: 4.843, interval_steps_per_second: 6.053, epoch: 2.0106[0m
[32m[2022-08-31 16:23:21,196] [    INFO][0m - loss: 0.61957884, learning_rate: 2.938888888888889e-05, global_step: 770, interval_runtime: 1.5463, interval_samples_per_second: 5.174, interval_steps_per_second: 6.467, epoch: 2.037[0m
[32m[2022-08-31 16:23:22,745] [    INFO][0m - loss: 0.74482732, learning_rate: 2.938095238095238e-05, global_step: 780, interval_runtime: 1.5483, interval_samples_per_second: 5.167, interval_steps_per_second: 6.459, epoch: 2.0635[0m
[32m[2022-08-31 16:23:24,288] [    INFO][0m - loss: 0.74435201, learning_rate: 2.9373015873015875e-05, global_step: 790, interval_runtime: 1.5439, interval_samples_per_second: 5.182, interval_steps_per_second: 6.477, epoch: 2.0899[0m
[32m[2022-08-31 16:23:25,822] [    INFO][0m - loss: 0.61089883, learning_rate: 2.9365079365079366e-05, global_step: 800, interval_runtime: 1.5332, interval_samples_per_second: 5.218, interval_steps_per_second: 6.522, epoch: 2.1164[0m
[32m[2022-08-31 16:23:25,823] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 16:23:25,823] [    INFO][0m -   Num examples = 1373[0m
[32m[2022-08-31 16:23:25,823] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 16:23:25,823] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 16:23:25,823] [    INFO][0m -   Total prediction steps = 43[0m
[32m[2022-08-31 16:23:37,526] [    INFO][0m - eval_loss: 2.1136062145233154, eval_accuracy: 0.5433357611070648, eval_runtime: 11.7027, eval_samples_per_second: 117.323, eval_steps_per_second: 3.674, epoch: 2.1164[0m
[32m[2022-08-31 16:23:37,527] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-800[0m
[32m[2022-08-31 16:23:37,527] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 16:23:39,289] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-800/tokenizer_config.json[0m
[32m[2022-08-31 16:23:39,289] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-800/special_tokens_map.json[0m
[32m[2022-08-31 16:23:45,605] [    INFO][0m - loss: 0.73565769, learning_rate: 2.9357142857142857e-05, global_step: 810, interval_runtime: 19.783, interval_samples_per_second: 0.404, interval_steps_per_second: 0.505, epoch: 2.1429[0m
[32m[2022-08-31 16:23:47,139] [    INFO][0m - loss: 0.82004423, learning_rate: 2.934920634920635e-05, global_step: 820, interval_runtime: 1.5335, interval_samples_per_second: 5.217, interval_steps_per_second: 6.521, epoch: 2.1693[0m
[32m[2022-08-31 16:23:48,697] [    INFO][0m - loss: 0.79988122, learning_rate: 2.9341269841269842e-05, global_step: 830, interval_runtime: 1.559, interval_samples_per_second: 5.132, interval_steps_per_second: 6.415, epoch: 2.1958[0m
[32m[2022-08-31 16:23:50,237] [    INFO][0m - loss: 0.73243818, learning_rate: 2.9333333333333333e-05, global_step: 840, interval_runtime: 1.5393, interval_samples_per_second: 5.197, interval_steps_per_second: 6.496, epoch: 2.2222[0m
[32m[2022-08-31 16:23:51,773] [    INFO][0m - loss: 0.84800062, learning_rate: 2.9325396825396827e-05, global_step: 850, interval_runtime: 1.536, interval_samples_per_second: 5.208, interval_steps_per_second: 6.51, epoch: 2.2487[0m
[32m[2022-08-31 16:23:53,312] [    INFO][0m - loss: 0.80736752, learning_rate: 2.9317460317460318e-05, global_step: 860, interval_runtime: 1.5392, interval_samples_per_second: 5.197, interval_steps_per_second: 6.497, epoch: 2.2751[0m
[32m[2022-08-31 16:23:54,858] [    INFO][0m - loss: 0.89100275, learning_rate: 2.930952380952381e-05, global_step: 870, interval_runtime: 1.5463, interval_samples_per_second: 5.174, interval_steps_per_second: 6.467, epoch: 2.3016[0m
[32m[2022-08-31 16:23:56,398] [    INFO][0m - loss: 0.60926223, learning_rate: 2.9301587301587303e-05, global_step: 880, interval_runtime: 1.5393, interval_samples_per_second: 5.197, interval_steps_per_second: 6.496, epoch: 2.328[0m
[32m[2022-08-31 16:23:57,940] [    INFO][0m - loss: 0.81681805, learning_rate: 2.9293650793650793e-05, global_step: 890, interval_runtime: 1.5427, interval_samples_per_second: 5.186, interval_steps_per_second: 6.482, epoch: 2.3545[0m
[32m[2022-08-31 16:23:59,475] [    INFO][0m - loss: 0.53452053, learning_rate: 2.9285714285714284e-05, global_step: 900, interval_runtime: 1.5349, interval_samples_per_second: 5.212, interval_steps_per_second: 6.515, epoch: 2.381[0m
[32m[2022-08-31 16:23:59,476] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 16:23:59,476] [    INFO][0m -   Num examples = 1373[0m
[32m[2022-08-31 16:23:59,476] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 16:23:59,476] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 16:23:59,476] [    INFO][0m -   Total prediction steps = 43[0m
[32m[2022-08-31 16:24:11,300] [    INFO][0m - eval_loss: 2.2992045879364014, eval_accuracy: 0.5207574654042243, eval_runtime: 11.8237, eval_samples_per_second: 116.123, eval_steps_per_second: 3.637, epoch: 2.381[0m
[32m[2022-08-31 16:24:11,301] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-900[0m
[32m[2022-08-31 16:24:11,301] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 16:24:13,206] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-900/tokenizer_config.json[0m
[32m[2022-08-31 16:24:13,207] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-900/special_tokens_map.json[0m
[32m[2022-08-31 16:24:19,506] [    INFO][0m - loss: 0.75335431, learning_rate: 2.9277777777777778e-05, global_step: 910, interval_runtime: 20.0308, interval_samples_per_second: 0.399, interval_steps_per_second: 0.499, epoch: 2.4074[0m
[32m[2022-08-31 16:24:21,055] [    INFO][0m - loss: 0.88276443, learning_rate: 2.9269841269841272e-05, global_step: 920, interval_runtime: 1.5484, interval_samples_per_second: 5.166, interval_steps_per_second: 6.458, epoch: 2.4339[0m
[32m[2022-08-31 16:24:22,604] [    INFO][0m - loss: 0.88667545, learning_rate: 2.9261904761904763e-05, global_step: 930, interval_runtime: 1.5488, interval_samples_per_second: 5.165, interval_steps_per_second: 6.457, epoch: 2.4603[0m
[32m[2022-08-31 16:24:24,153] [    INFO][0m - loss: 0.60428009, learning_rate: 2.9253968253968257e-05, global_step: 940, interval_runtime: 1.5495, interval_samples_per_second: 5.163, interval_steps_per_second: 6.454, epoch: 2.4868[0m
[32m[2022-08-31 16:24:25,700] [    INFO][0m - loss: 0.95294142, learning_rate: 2.9246031746031748e-05, global_step: 950, interval_runtime: 1.5472, interval_samples_per_second: 5.171, interval_steps_per_second: 6.463, epoch: 2.5132[0m
[32m[2022-08-31 16:24:27,250] [    INFO][0m - loss: 0.80352631, learning_rate: 2.923809523809524e-05, global_step: 960, interval_runtime: 1.5503, interval_samples_per_second: 5.16, interval_steps_per_second: 6.45, epoch: 2.5397[0m
[32m[2022-08-31 16:24:28,790] [    INFO][0m - loss: 0.82687292, learning_rate: 2.9230158730158733e-05, global_step: 970, interval_runtime: 1.5393, interval_samples_per_second: 5.197, interval_steps_per_second: 6.496, epoch: 2.5661[0m
[32m[2022-08-31 16:24:30,336] [    INFO][0m - loss: 0.74503865, learning_rate: 2.9222222222222224e-05, global_step: 980, interval_runtime: 1.5457, interval_samples_per_second: 5.175, interval_steps_per_second: 6.469, epoch: 2.5926[0m
[32m[2022-08-31 16:24:31,876] [    INFO][0m - loss: 0.69472098, learning_rate: 2.9214285714285715e-05, global_step: 990, interval_runtime: 1.5408, interval_samples_per_second: 5.192, interval_steps_per_second: 6.49, epoch: 2.619[0m
[32m[2022-08-31 16:24:33,422] [    INFO][0m - loss: 0.90430794, learning_rate: 2.9206349206349206e-05, global_step: 1000, interval_runtime: 1.5454, interval_samples_per_second: 5.177, interval_steps_per_second: 6.471, epoch: 2.6455[0m
[32m[2022-08-31 16:24:33,423] [    INFO][0m - ***** Running Evaluation *****[0m
[32m[2022-08-31 16:24:33,423] [    INFO][0m -   Num examples = 1373[0m
[32m[2022-08-31 16:24:33,423] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 16:24:33,423] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 16:24:33,423] [    INFO][0m -   Total prediction steps = 43[0m
[32m[2022-08-31 16:24:45,389] [    INFO][0m - eval_loss: 2.3379251956939697, eval_accuracy: 0.5142024763292061, eval_runtime: 11.9658, eval_samples_per_second: 114.744, eval_steps_per_second: 3.594, epoch: 2.6455[0m
[32m[2022-08-31 16:24:45,390] [    INFO][0m - Saving model checkpoint to ./checkpoints/checkpoint-1000[0m
[32m[2022-08-31 16:24:45,390] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 16:24:47,366] [    INFO][0m - tokenizer config file saved in ./checkpoints/checkpoint-1000/tokenizer_config.json[0m
[32m[2022-08-31 16:24:47,366] [    INFO][0m - Special tokens file saved in ./checkpoints/checkpoint-1000/special_tokens_map.json[0m
[32m[2022-08-31 16:24:52,179] [    INFO][0m - 
Training completed. 
[0m
[32m[2022-08-31 16:24:52,479] [    INFO][0m - Loading best model from ./checkpoints/checkpoint-600 (score: 0.5593590677348871).[0m
[32m[2022-08-31 16:24:53,074] [    INFO][0m - train_runtime: 339.3261, train_samples_per_second: 891.178, train_steps_per_second: 111.397, train_loss: 1.5000194416046142, epoch: 2.6455[0m
[32m[2022-08-31 16:24:53,075] [    INFO][0m - Saving model checkpoint to ./checkpoints/[0m
[32m[2022-08-31 16:24:53,076] [    INFO][0m - Trainer.model is not a `PretrainedModel`, only saving its state dict.[0m
[32m[2022-08-31 16:24:54,978] [    INFO][0m - tokenizer config file saved in ./checkpoints/tokenizer_config.json[0m
[32m[2022-08-31 16:24:54,979] [    INFO][0m - Special tokens file saved in ./checkpoints/special_tokens_map.json[0m
[32m[2022-08-31 16:24:54,981] [    INFO][0m - ***** train metrics *****[0m
[32m[2022-08-31 16:24:54,981] [    INFO][0m -   epoch                    =     2.6455[0m
[32m[2022-08-31 16:24:54,981] [    INFO][0m -   train_loss               =        1.5[0m
[32m[2022-08-31 16:24:54,981] [    INFO][0m -   train_runtime            = 0:05:39.32[0m
[32m[2022-08-31 16:24:54,981] [    INFO][0m -   train_samples_per_second =    891.178[0m
[32m[2022-08-31 16:24:54,981] [    INFO][0m -   train_steps_per_second   =    111.397[0m
[32m[2022-08-31 16:24:54,987] [    INFO][0m - ***** Running Prediction *****[0m
[32m[2022-08-31 16:24:54,987] [    INFO][0m -   Num examples = 1749[0m
[32m[2022-08-31 16:24:54,987] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 16:24:54,987] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 16:24:54,987] [    INFO][0m -   Total prediction steps = 55[0m
[32m[2022-08-31 16:25:10,124] [    INFO][0m - ***** test metrics *****[0m
[32m[2022-08-31 16:25:10,125] [    INFO][0m -   test_accuracy           =     0.5506[0m
[32m[2022-08-31 16:25:10,125] [    INFO][0m -   test_loss               =     1.8956[0m
[32m[2022-08-31 16:25:10,125] [    INFO][0m -   test_runtime            = 0:00:15.13[0m
[32m[2022-08-31 16:25:10,125] [    INFO][0m -   test_samples_per_second =    115.543[0m
[32m[2022-08-31 16:25:10,125] [    INFO][0m -   test_steps_per_second   =      3.633[0m
[32m[2022-08-31 16:25:10,125] [    INFO][0m - ***** Running Prediction *****[0m
[32m[2022-08-31 16:25:10,125] [    INFO][0m -   Num examples = 2600[0m
[32m[2022-08-31 16:25:10,126] [    INFO][0m -   Pre device batch size = 32[0m
[32m[2022-08-31 16:25:10,126] [    INFO][0m -   Total Batch size = 32[0m
[32m[2022-08-31 16:25:10,126] [    INFO][0m -   Total prediction steps = 82[0m
[32m[2022-08-31 16:25:39,484] [    INFO][0m - Predictions for iflytekf saved to ./fewclue_submit_examples.[0m
run.sh: line 65: --model_name_or_path: command not found
