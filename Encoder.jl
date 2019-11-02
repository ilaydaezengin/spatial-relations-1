MAX_LENGTH = 60
EMBEDDING_SIZE = 64

function encode(batchofenvs)
    images, captions, cameras = getdata(batchofenvs)
    
    # caption handling
    caption_embeds = arrange(captions, vocab) # pad or cut to a specified length of words, perform embedding
    captions_hat = CNN(caption_embeds) # d i ^
    
    # camera handling
    cameras_hat = MLP(cos(cameras),sin(cameras))
    
    # create hiddens vectors
    hiddens = MLP(captions_hat, cameras_hat) # [h1, h2, .., h10]
    r = aggregate(hiddens)
    return r
end


# provides captions and cameras from the specified env
function getdata(batchofenvs)
    images = batchofenvs["images"]
    captions = batchofenvs["captions"]
    cameras = batchofenvs["cameras"]
    return images, captions, cameras
end

# captions: 10 captions for an env/scene
# vocab: dictionary of known words
# pads or cuts the sentence to a specified length of words, performs embeddings
# output_size = 10-el array of 64xMAX_LENGTH
function arrange(batchofcaptions, vocab)
    int_batchofcaptions = []
    for i in 1:BATCH_SIZE
        int_captions = []
        captions = batchofcaptions[i,:]
        for c in captions
            int_cap = []
            str = split(c, " ")
            for i in 1:MAX_LENGTH
               if i <= length(str)
                    push!(int_cap, get!(vocab, str[i], KnetArray(Knet.xavier(EMBEDDING_SIZE))))
                else
                    push!(int_cap, get!(vocab, "", vocab[""])) # padding, same embedding with whitespace
                end
            end
            push!(int_captions, hcat(int_cap...))
        end
        push!(int_batchofcaptions, int_captions)
    end
    return int_batchofcaptions
end


# performs convolution operation through a neural network
function CNN(caption_embeds)
    return captions_hat
end

# multilayer perceptron
function MLP(a,b) # or concat?
   return c
end

# sums up all hidden vectors and divide it by the number of them
function aggregate(hiddens)
    sum = []
    for h in hiddens
        sum = sum + h
    end
    r = sum/length(hiddens)
    return r
end
