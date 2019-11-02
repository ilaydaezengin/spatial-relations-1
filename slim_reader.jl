
module Slim_Reader
using PyCall
using Knet

push!(pyimport("sys")["path"], pwd());
READER = pyimport(:reader)

function get_train_filenames(slim_dataset_path, include_synthetic_data=true, include_turk_data=false)
    get_filenames(slim_dataset_path, "train", include_synthetic_data, include_turk_data)
end

function get_test_filenames(slim_dataset_path, include_synthetic_data=true, include_turk_data=false)
    get_filenames(slim_dataset_path, "test", include_synthetic_data, include_turk_data)
end

function get_validation_filenames(slim_dataset_path, include_synthetic_data=true, include_turk_data=false)
    get_filenames(slim_dataset_path, "valid", include_synthetic_data, include_turk_data)
end

function get_filenames(slim_dataset_dir::String, dataset_type="train", include_synthetic_data=true, include_turk_data=false)
    file_names = Array{String}([])
    if include_synthetic_data
        synthetics = slim_dataset_dir*"/synthetic_data/"*dataset_type*"/"
        dirs = readdir(synthetics)
        for dir in dirs
            full_dir = synthetics*dir
            if isfile(full_dir)
                push!(file_names, full_dir)
            elseif isdir(full_dir)
                full_dir = full_dir*"/"*first(readdir(full_dir))
                isfile(full_dir) && push!(file_names, full_dir)
            end
        end
    end

    if include_turk_data
        full_dir = slim_dataset_dir*"/turk_data/"*dataset_type*".tfrecord"
        isfile(full_dir) && push!(file_names, full_dir)
    end
    return file_names
end

function make_dataset_object(file_names, batchsize)
    filenames, iterator, next_element = READER.make_dataset(batch_size=batchsize)
    dataset = Dict("filenames" => filenames, "iterator"=>iterator,
                   "next_element"=>next_element)

    sess = READER.tf.Session()
    sess.run(iterator.initializer, feed_dict=Dict{}(filenames => file_names))
    dataset["sess"] = sess
    return dataset
end

function get_next_chunk(dataset)
    session = dataset["sess"]
    next_element_func = dataset["next_element"]
    return session.run(next_element_func)
end

function get_next_batch(dataset, atype=Array{Float32}, add_top_views=false, add_camera_coords=false)
    if add_top_views && add_camera_coords
        error("top view images do not have camera coordinate info")
    end

    chunk = get_next_chunk(dataset)
    chunk_dict = chunk[3]
    batch = Dict()

    # images: B, V, W, W, C  where B: batchsize, V: number of views, W: width, C: number of channels
    images = chunk_dict["images"]
    # captions: B, V where B: batchsize, V: number of views
    captions = chunk_dict["captions"]
    if add_top_views
        top_views = chunk_dict["top_down"]
        top_views = reshape(top_views ,(first(size(top_views)), 1, size(top_views)[2:end]...))
        images = cat(images, top_views, dims=2);

        # since no caption is present for top-view, sample one from others.
        rand_captions = captions[:, rand(1:size(captions)[end])]
        rand_captions = reshape(rand_captions, length(rand_captions), 1)
        captions = cat(captions, rand_captions, dims=2)
    end

    images = convert(atype, images./255);
    batch["images"] = images

    captions = convert(Array{String}, captions)
    batch["captions"] = captions

    # cameras: B, V, C where B: batchsize, V: number of views, W: width, C: camera coordiates in 3D
    add_camera_coords && (batch["cameras"] = convert(atype, chunk_dict["cameras"]))

    return batch
end

end #module
