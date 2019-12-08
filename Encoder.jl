MAX_LENGTH = 60
EMBEDDING_SIZE = 64
NUM_CAPTIONS_PER_SCENE = 10

#=
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

# performs convolution operation through a neural network
function CNN(caption_embeds)
    return captions_hat
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

=#

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






function aggregate(h) # batch of h s
    r = []
    for i in 1:BATCH_SIZE
        push!(r, sum(h[:,10*(i-1)+1:10*i-1], dims=2)./(NUM_CAPTIONS_PER_SCENE-1)) # 1...9, 11...19, ...
    end
    return hcat(r...)
end

function getembed(c::Corpus, w)
    if haskey(c.vocabid, w)
        return value(c.vocab)[:,c.vocabid[w]]
    else
        println("There is no such word in the vocab!!!!")
    end
end

function updatevocabs!(c::Corpus, w)
    if !haskey(c.vocabid, w)
        if size(c.vocab,2) == length(c.vocabid)
            #println("Vocab and vocabid are in order!")
            c.vocabid[w] = length(c.vocabid)+1
            c.vocab = param(hcat(value(c.vocab), KnetArray{Float32}(Knet.xavier(EMBEDDING_SIZE))))
        end
    end    
end

function arrange(batchofcaptions, co::Corpus)
    int_batchofcaptions = []
    for i in 1:BATCH_SIZE
        int_captions = []
        captions = batchofcaptions[i,:]
        for c in captions
            int_cap = []
            str = split(c, " ")
            for i in 1:MAX_LENGTH
               if i <= length(str)
                    updatevocabs!(co, str[i])
                    push!(int_cap, getembed(co, str[i])) #KnetArray(Knet.xavier(EMBEDDING_SIZE))
                else
                    push!(int_cap, getembed(co, "")) # padding, same embedding with whitespace
                end
            end
            push!(int_captions, hcat(int_cap...))
        end
        push!(int_batchofcaptions, int_captions)
    end
    return int_batchofcaptions
end

function build_angles(tuplebatch)
    output = []
    for i in 1:BATCH_SIZE
        for j in 1:NUM_CAPTIONS_PER_SCENE
            sin_value = tuplebatch[i,j,3]/sqrt(tuplebatch[i,j,1]^2+tuplebatch[i,j,3]^2)
            cos_value = tuplebatch[i,j,1]/sqrt(tuplebatch[i,j,1]^2+tuplebatch[i,j,3]^2)
            push!(output, KnetArray([sin_value, cos_value])) 
        end
    end
    
    return cat(output..., dims=2)
end

function createconvinput(caption_embeds)
    o = []
    for i in 1:BATCH_SIZE
        push!(o, cat(transpose(caption_embeds[i])...,dims=3))
    end
    return o = cat(o..., dims=3)
end


function createimgencinput(images) # images size: 50×10×128×128×3
    out = []
    for i in 1:BATCH_SIZE
        push!(out, permutedims(images[i,1:9,:,:,:], (2,3,4,1)))
    end
    return cat(out..., dims=4)
end


function creategoldimg(images)
    unseen_img = pool(permutedims(images[:,10,:,:,:], (2,3,4,1)), window=4, stride=4) # shall the batchsize be in the end
    return unseen_img
end