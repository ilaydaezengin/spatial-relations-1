# include("Encoder.jl") problem?

mutable struct MultilayerPerceptron
    layers
    MultilayerPerceptron(layers...) = new(layers)
end
(m::MultilayerPerceptron)(x) = (for l in m.layers; x = l(x); end; x)

struct Layer0; w; b; end
Layer0(ir::Int, ic::Int, o::Int) = Layer0(param(o,ir),param0(o, ic))
(l::Layer0)(x) = (l.w * x .+ l.b)

mutable struct EmbedModel
    w
end

function EmbedModel()
    dim1 = 30
    dim2 = EMBEDDING_SIZE
    dim3 = 32
    w = reshape(KnetArray(Knet.xavier(dim1*dim2*dim3)), (1,dim1,dim2,dim3))
    return EmbedModel(w)
end

function (e::EmbedModel)(x)
    output = conv4(e.w, x, dilation=2)
    return output
end

mutable struct CaptionEncoder
    embed_model
end

function CaptionEncoder()
   embed_model = EmbedModel()
   return CaptionEncoder(embed_model)
end

function (c::CaptionEncoder)(captions, vocabid, vocab)
    vocabid, vocab, caption_embeds = arrange(captions, vocabid, vocab)
    input = createconvinput(caption_embeds)
    input = reshape(input, (1, MAX_LENGTH, EMBEDDING_SIZE, BATCH_SIZE*NUM_CAPTIONS_PER_SCENE))
    di_hat = c.embed_model(input)
    captions_hat = convert(KnetArray{Float32}, reshape(reshape(di_hat, (2, 32, BATCH_SIZE*NUM_CAPTIONS_PER_SCENE)), (EMBEDDING_SIZE, BATCH_SIZE*NUM_CAPTIONS_PER_SCENE)))
    return captions_hat, vocabid, vocab
end

mutable struct AngleEncoder
    mlp_model
end

function AngleEncoder()
    dim1 = 2 # cos and sin
    dim2 = BATCH_SIZE*NUM_CAPTIONS_PER_SCENE
    dim3 = 32 # from paper
    mlp1=MultilayerPerceptron(Layer0(dim1, dim2, dim3)) # MLP1 dimensionality 32
    return AngleEncoder(mlp1)
end

function (a::AngleEncoder)(cameras)
    tuples = build_angles(cameras)
    cameras_hat = a.mlp_model(tuples)
    return cameras_hat
end

mutable struct ImageConvModel
    w
end

function ImageConvModel()
    dim1 = 17
    dim2 = 17
    dim3 = 3 # RGB
    w = reshape(KnetArray{Float32}(Knet.xavier(dim1*dim2*dim3)), (dim1,dim2,dim3,1))
    return ImageConvModel(w)
end

function (i::ImageConvModel)(x)
    output = conv4(i.w, x)
    return output
end

mutable struct ImageEncoder
    image_conv_model
    sampling_model
end

function ImageEncoder()
    image_conv_model = ImageConvModel()
    sampling_model = MultilayerPerceptron(Layer0(256,450,128), Layer0(128, 450, 18))
    return ImageEncoder(image_conv_model, sampling_model)
end

function (i::ImageEncoder)(images)
    imgencoderin = createimgencinput(images)
    imgencoderin2 = pool(imgencoderin, window=4, stride=4)
    himg = i.image_conv_model(imgencoderin2)
    himg = himg[:,:,1,:]
    himg = reshape(himg, (256,450))
    z = reshape(i.sampling_model(himg), (162, 50))
    return z
end

mutable struct RepresentationModel
    caption_encoder
    angle_encoder
    image_encoder
    mlp_model #mlp2 that takes concatenated di_hat and ci_hat, output = hi_hat
end

function RepresentationModel()
    caption_encoder = CaptionEncoder()
    angle_encoder = AngleEncoder()
    image_encoder = ImageEncoder()
    mlp_model = MultilayerPerceptron(Layer0(96, 500, 128), Layer0(128, 500, 196), Layer0(196, 500, 256)) # MLP2 dimensionality 256
   return RepresentationModel(caption_encoder, angle_encoder, image_encoder, mlp_model) 
end

function (re::RepresentationModel)(images, captions, cameras, vocabid, vocab)
    captions_hat, vocabid, vocab = re.caption_encoder(captions, vocabid, vocab)
    cameras_hat = re.angle_encoder(cameras)
    
    unseen_ang = []
    for i in 1:size(cameras_hat, 2)
        if mod(i,10) == 0
           push!(unseen_ang, cameras_hat[:,i]) 
        end
    end
    unseen_ang = hcat(unseen_ang...)
    
    h = re.mlp_model(cat(captions_hat, cameras_hat, dims=1))
    r = aggregate(h)
    z = re.image_encoder(images)
    
    return r, z, unseen_ang, vocabid, vocab
end

representationModel = RepresentationModel()
r, z, unseen_ang, vocabid, vocab = representationModel(aimages, acaptions, acameras, vocabid, vocab)

mutable struct GenerationModel
    w
end

function GenerationModel()
    dim1 = 32
    dim2 = 32
    dim3 = 3 # RGB
    dim4 = 450
    w = reshape(KnetArray{Float32}(Knet.xavier(dim3*dim1*dim2*dim4)), (dim1,dim2,dim3,dim4))
    return GenerationModel(w)
end

function (g::GenerationModel)(x)
    final_output = deconv4(g.w, x)
    return final_output
end

mutable struct Network
    representationModel
    generationModel
end

function Network()
    representationModel = RepresentationModel()
    generationModel = GenerationModel()
    return Network(representationModel, generationModel)
end

function (n::Network)(images, captions, cameras, vocabid, vocab)
    r, z, unseen_ang, vocabid, vocab = n.representationModel(images, captions, cameras, vocabid, vocab)
    gen_input = reshape(vcat(z, vcat(r, unseen_ang)),(1,1,450,50)) # decoder input
    output = n.generationModel(gen_input)
    return output, vocabid, vocab
end


