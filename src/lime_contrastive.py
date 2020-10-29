weights21= [i / 200 for i in range(1, 200)]
weights22= [1 + i / 200 for i in range(1, 200)]
#weights22 = [-1 - (i / 140) for i in range(1, 140)]
#weights2 = [-i / 140 for i in range(1, 140)]
weights2 =   weights21  + weights22

weights1 = [1- i/100 for i in range(1,60)]
weights11 = [1 + i/100 for i in range(1,60)]
weights1 = weights1 + weights11
weights1 = [1]
segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.22)
for ii, tindex in enumerate(my_indices):
    #if ii < 1:
    #    continue
    vinput1, label = get_input(test_loader, tindex)
    pic_name = "index_" + str(tindex) + "_label_" + str(label.cpu().data.numpy().squeeze()) + "_samples_" + str(num_samples)
    pic_dest = os.path.join(save_dir, pic_name + file_suffix)
    pic_dest_simple = os.path.join(save_dir, pic_name + '_simple_' + file_suffix)
    print(pic_dest)
    image = vinput1
    np_image = image.cpu()[0][0].numpy()
    #probs = pred_func1(image).squeeze()
    lbl = 8  # predictions[1]
    probs = pred_func(np_image).squeeze()
    aa = probs[label.item()]
    bb = probs[lbl]
    predictions = probs.argsort()[::-1]
    pred_probs = np.sort(probs)[::-1]
    pred_probs[0] = aa
    pred_probs[1] = bb
    exptop2 = explainer.explain_instance(np_image, pred_func,
                                                                 hide_color=True, num_samples=num_samples,
                                                                 top_labels=10
                                         , segmentation_fn=segmenter
                                         )
    a,b  = exptop2.get_image_and_mask(label.item(),positive_only=True,num_features=100000, hide_rest=True)
    b = b.astype('float')
    a = (a - a.min()) / (a.max() - a.min() + 0.000001)
    c,d  = exptop2.get_image_and_mask(lbl,positive_only=True,num_features=100000, hide_rest=True)
    c = (c - c.min()) / (c.max() - c.min() + 0.000001)
    d = d.astype('float')
    plt.close()
    print(label.item(), ' , ', lbl)
    print(pred_probs[0], ' , ', pred_probs[1])
    ch1 = 0
    ch2 = 0
    avg_pic = Tensor(64,64,3).fill_(0)
    wcnt = 0
    for we1 in weights1:
        for we2 in weights2:
            #apc = rgb2grey(we1 * a + we2 *c)
            apc = -we1 * a[:,:,0] + we2 * c[:,:,0]
            apc = Tensor(apc)
            #avg_pic += apc.clone()
            apc = apc.unsqueeze(0).unsqueeze(0)
            apc = make_in_range(apc + vinput1.clone(), vinput1.min(), vinput1.max())
            yapc = D(apc)
            val1 = SoftMax(yapc)[0, label.item()].item()
            val2 = SoftMax(yapc)[0, lbl].item()
            #break
            if pred_probs[0] - val1 < 0.03 or val2 - pred_probs[1] < 0.03:# or val1 < 0.15 or val2 < 0.15:
                continue
            avg_pic += Tensor(we1 * -a + we2 * c)
            ch1 += val1 - pred_probs[0]
            ch2 += val2 - pred_probs[1]
            wcnt += 1
    lst_vals[0].append(ch1/wcnt)
    lst_vals[1].append(ch2 / wcnt)
    plt.close()
    fig = plt.figure()
    ee = avg_pic/ wcnt
    #ee = make_in_range(ee, 0, 1)
    ee = ee[:,:,0].clone() #+ vinput1.clone()
    print('ee max and min: ', ee.max().item(), ', ', ee.min().item())
    rev_inp = vinput1.clone() *0.5 + 0.5
    rev_inp = rev_inp.squeeze(0).squeeze(0)
    eec = make_in_range(ee, vinput1.min().item(), vinput1.max().item())
    eec = ee
    exp_img = get_overlayed_image(rev_inp.detach().cpu().numpy(),  eec.detach().cpu().numpy())
    plt.imshow(exp_img)
    frame = plt.gca()
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])
    #plt.savefig('abcd.jpg', bbox_inches='tight',transparent=True, pad_inches=0)
    plt.savefig(pic_dest, bbox_inches='tight',transparent=True, pad_inches=0)
    plt.close()
    apc = 1 * a[:,:,0] + 1 * c[:,:,0]
    apc = Tensor(apc)
