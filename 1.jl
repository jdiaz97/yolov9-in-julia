import ONNXRunTime as ONNX
using Images, FileIO 
using ImageDraw 

const model::ONNXRunTime.InferenceSession = ONNX.load_inference("yolov9-c.onnx", execution_provider=:cpu)

function get_input_array(img)
    resized = imresize(img, (640, 640))
    mat = channelview(resized)
    original_array = Float32.(mat)
    return reshape(original_array, (1,size(original_array)...))
end

function run_inference(img)
    input_array = get_input_array(img)
    res = model(Dict("images" => input_array) )["output0"] ## must be 1x84x8400
    return res 
end

function process_output(res)
    ## Processing Outputs
    predictions = transpose(res[1, : , :]) ## must be (8400, 84)
    conf_threshold = 0.8
    scores = maximum(predictions[:, 5:end], dims=2)

    ## Filter based on confidence
    scores[scores .> conf_threshold]
    predictions = predictions[vec(scores .> conf_threshold),:]

    class_ids = argmax(predictions[:, 5:end], dims=2)
    class_ids = map(i->i[2], class_ids)

    boxes = predictions[:, begin:4]
    return boxes
end

function yolobox(x,y,w,h)
    x1, y1 = x-w/2, y-h/2
    x2, y2 = x+w/2, y+h/2
    return trunc.(Int,[x1, y1, x2, y2])
end

function drawbox(img,x1,y1,x2,y2)
    draw(imresize(img, (640, 640)), Polygon(RectanglePoints(Point(x1,y1), Point(x2, y2))), RGB{N0f8}(1))
end

img = load("ñaña.jpg")
res = run_inference(img)
boxes = process_output(res)

x1,y1,x2,y2 = yolobox(boxes[1,1],boxes[1,2],boxes[1,3],boxes[1,4]) ## First pred
img_draw = drawbox(img,x1,y1,x2,y2)
img_draw

# save("pred.jpg",img_draw)
