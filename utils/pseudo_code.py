for step in range(all_step):
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step, init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step, init_lr=args.lr)

        zero_grad_all()        
        pseudo_labels, mask_loss = do_fixmatch(data_t_unl,F1,G,thresh,criterion_pseudo)
        
        f_batch_source, feat_dict_source = update_features(feat_dict_source, data_s, G, 0, source = True)

        #if step >=0 and step % 250 == 0 and step<=3500:
        if step>=2000:
            if step % 1000 == 0:
                _ = do_source_weighting(target_loader_misc,feat_dict_source, G, K_farthest_source, per_class_raw = raw_weights_to_pass, weight=1, aug = 2, phi = 0.5, only_for_poor=True, poor_class_list=poor_class_list, weighing_mode='N',weigh_using=weigh_using)

        if step >=8000:
            do_lab_target_loss(G,F1,data_t,im_data_t, gt_labels_t, criterion_lab_target)

        #output = G(data)
        output = f_batch_source
        out1 = F1(output)

        if step>=2000:
            names_batch = list(data_s[2])
            idx = [feat_dict_source.names.index(name) for name in names_batch] 
            weights_source = feat_dict_source.sample_weights[idx].cuda()
            loss = torch.mean(weights_source * criterion(out1, target))
        else:
            loss = torch.mean(criterion(out1, target))
        
        loss.backward(retain_graph=True)

        output = G(im_data_tu)
        loss_t = adentropy(F1, output, args.lamda)
        loss_t.backward()

        _, acc_labeled_target, _, per_cls_acc = test(target_loader, mode = 'Labeled Target')
        _, acc_test,_,_ = test(target_loader_test, mode = 'Test')
        _, acc_val, _, _ = test(target_loader_val, mode = 'Val')
