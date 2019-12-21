using Knet

mutable struct Corpus
    vocabid
    vocab
end

function Corpus()
    vocabid = Dict()
    vocabid = Dict("" => 1) # both padding and whitespace KnetArray(zeros(EMBEDDING_SIZE))
    vocab = param(reshape(KnetArray{Float32}(zeros(EMBEDDING_SIZE)),(EMBEDDING_SIZE,1)))
    return Corpus(vocabid,vocab)
end


mutable struct MultilayerPerceptron
    layers
    MultilayerPerceptron(layers...) = new(layers)
end
(m::MultilayerPerceptron)(x) = (for l in m.layers; x = l(x); end; x)


struct Layer0; w; b; end
Layer0(ir::Int, ic::Int, o::Int) = Layer0(param(o,ir),param0(o, ic))
(l::Layer0)(x) = (l.w * x .+ l.b)


struct ConvModel; w; b; f; end
(c::ConvModel)(x) = c.f.(conv4(c.w, x, dilation=2) .+ c.b)
ConvModel(w1,w2,cx,cy,f=relu) = ConvModel(param(w1,w2,cx,cy), param0(1,2,cy,1), f)


mutable struct CaptionEncoder
    conv_model
    corpus
end
function CaptionEncoder()
    dim1 = 30
    dim2 = 64 #EMBEDDING_SIZE
    dim3 = 32
    conv_model = ConvModel(1,dim1,dim2,dim3)
    corpus = Corpus()
    return CaptionEncoder(conv_model, corpus)
end
function (c::CaptionEncoder)(captions)
    caption_embeds = arrange(captions, c.corpus)
    input = createconvinput(caption_embeds)
    input = reshape(input, (1, MAX_LENGTH, EMBEDDING_SIZE, BATCH_SIZE*NUM_CAPTIONS_PER_SCENE))
    di_hat = c.conv_model(input)
    captions_hat = reshape(reshape(di_hat, (2, 32, BATCH_SIZE*NUM_CAPTIONS_PER_SCENE)), (EMBEDDING_SIZE, BATCH_SIZE*NUM_CAPTIONS_PER_SCENE))
    return captions_hat
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


mutable struct ImageConvModel; w; b; f; end
(i::ImageConvModel)(x) = i.f.(conv4(i.w, x) .+ i.b)
ImageConvModel(w1,w2,cx,cy,f=relu) = ImageConvModel(param(w1,w2,cx,cy), param0(16,16,cy,1), f)


mutable struct SamplingModel
    mu_layer
    logsigma_layer
end
function SamplingModel()
    mu_layer = Layer0(256,50,162)
    logsigma_layer = Layer0(256, 50, 162)
    return SamplingModel(mu_layer, logsigma_layer)
end
function (s::SamplingModel)(himg)
    mu = s.mu_layer(himg)
    #mu = reshape(mu, (162,50))
    logsigma = s.logsigma_layer(himg)
    #logsigma = reshape(logsigma, (162,50))
    sigma = exp.(logsigma)
    noise = randn!(similar(mu)) #162x50
    z = mu .+ noise .* sigma
    return (z, mu, sigma, logsigma) # z 162x50
end


mutable struct RepresentationModel
    caption_encoder
    angle_encoder
    #image_encoder
    mlp_model #mlp2 that takes concatenated di_hat and ci_hat, output = hi_hat
end
function RepresentationModel()
    caption_encoder = CaptionEncoder()
    angle_encoder = AngleEncoder()
    #image_encoder = ImageEncoder()
    mlp_model = MultilayerPerceptron(Layer0(96, 500, 128), Layer0(128, 500, 196), Layer0(196, 500, 256)) # MLP2 dimensionality 256
   return RepresentationModel(caption_encoder, angle_encoder, mlp_model) # image_encoder, 
end
function (re::RepresentationModel)(captions, cameras) #images, 
    captions_hat = re.caption_encoder(captions)
    cameras_hat = re.angle_encoder(cameras)
    
    # move!
    unseen_ang = []
    for i in 1:size(cameras_hat, 2)
        if mod(i,10) == 0
           push!(unseen_ang, cameras_hat[:,i]) 
        end
    end
    unseen_ang = hcat(unseen_ang...)
    
    h = re.mlp_model(cat(captions_hat, cameras_hat, dims=1))
    r = aggregate(h)
    # z = re.image_encoder(images)
    
    return r, unseen_ang # z,
end


mutable struct ImageEncoder
    cv_linear_model
    image_conv_model
    sampling_model
end
function ImageEncoder()
    cv_linear_model = Layer0(288, 50, 1024)
    image_conv_model = ImageConvModel(17, 17, 4, 1)
    sampling_model = SamplingModel()
    return ImageEncoder(cv_linear_model, image_conv_model, sampling_model)
end
function (i::ImageEncoder)(images, cv)
    imgencoderin1 = creategoldimg(images) # 32x32x3x50
    imgencoderin2 = cat(imgencoderin1, reshape(i.cv_linear_model(cv), (32, 32, 1, 50)), dims=3)
    himg = i.image_conv_model(imgencoderin2)
    himg = himg[:,:,1,:]
    himg = reshape(himg, (256,50))
    z, mu, sigma, logsigma = i.sampling_model(himg)
    return (z, mu, sigma, logsigma)
end


mutable struct ImageDecoder; w; b; f; end
(im::ImageDecoder)(x) = relu.(deconv4(im.w, x) .+ im.b)
ImageDecoder() = ImageDecoder(param(32,32,3,450), param0(32,32,3,1), relu)


mutable struct GenerationModel
    image_encoder
    image_decoder
end
function GenerationModel()
    image_encoder = ImageEncoder()
    image_decoder = ImageDecoder()
    return GenerationModel(image_encoder, image_decoder)
end
function (ge::GenerationModel)(images, conditional_variable)
    z, mu, sigma, logsigma = ge.image_encoder(images, conditional_variable)
    gen_input = reshape(vcat(z, conditional_variable),(1,1,450,50)) # decoder input
    rimages = ge.image_decoder(gen_input)
    return rimages, mu, sigma, logsigma
end


mutable struct Output
    rimages
    mu
    sigma
    logsigma
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
function (n::Network)(images, captions, cameras)
    r, unseen_ang = n.representationModel(captions, cameras) #images,  z,
    conditional_variable = vcat(r, unseen_ang)
    rimages, mu, sigma, logsigma = n.generationModel(images, conditional_variable)
    return Output(rimages, mu, sigma, logsigma)
end