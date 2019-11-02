function train(train_data, )
    for env in train_data
        r = encode(env)
        im_pred = generate(thetas, r)
        # thetas = 10 element array of tuples [x,y,z], im_pred = 10-el array of matrices/predicted images
        l = loss(im_pred, im_gold)
        updateweights!(l, )
    end
end