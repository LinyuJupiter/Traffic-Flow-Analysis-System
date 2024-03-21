clear;
clc;
%% 背景处理与视频读取区
pathnums='001';
% vid=VideoReader(['视频数据集\',pathnums,'.mp4']);
% background=zeros(vid.Height,vid.Width,3);
% for i=1:500
%     background=background+im2double(read(vid,i));
% end
% background=background/i;
% save([pathnums,'.mat'],"background","vid")
load([pathnums,'.mat'])
%% 计算区
% 获取图像大小
issave=0;%是否保存GIF图
[height, width, ~] = size(background);
fig=figure;
rect_handles = cell(0); %存放矩形框
first=1;% 初始帧
for times=first:400
    video=im2double(read(vid,times)); %当前帧
    differ=abs(video-background); %作差
    gray_img=rgb2gray(differ); %灰度化
    level=graythresh(gray_img); %全局阈值
    bm_img=imbinarize(gray_img,level-0.025); %二值化
    se = strel('rectangle', [5, 5]);
    open_img=imopen(bm_img,se); %开运算
    bw_img_cleaned=bwareaopen(open_img,300);% 去除小面积
    close_img=imclose(bw_img_cleaned,se); %闭运算
    close_img(280:1080,1720:1920)=0;%去除右下角
    SE=strel('disk',13);
    swell_img=imdilate(close_img,SE); %膨胀13像素
    % 获取连通域信息
    [L, num] =bwlabel(swell_img,8);
    stats = regionprops('table',L, 'basic');
    areas = [stats.Area];
    idxs = find(areas > 2500); %选择面积较大的连通域
    % 创建一个与原图像大小相同的全黑图像
    rect = zeros(height, width);
    % 遍历所有连通域，将其对应的位置在新图像中填充为白色
    if times==first
        h=imshow(video);% 第一次循环则显示图片
        for i = 1:length(idxs)
            bbox = round(stats.BoundingBox(idxs(i),:));
            rect(bbox(2):(bbox(2)+bbox(4)-1), bbox(1):(bbox(1)+bbox(3)-1)) = 1;
            rect_handles{end+1}=rectangle('Position', bbox, 'LineWidth', 2, 'EdgeColor', 'r');
        end
    else
        set(h,"CData",video)% 之后的循环都只修改图片显示源
        % 更新所有矩形框的位置和大小，并动态创建或删除矩形框对象
        for p=1:size(rect_handles,2)
            delete(rect_handles{p});
            rect_handles{p} = [];
        end
        for i = 1:length(idxs)
            bbox = round(stats.BoundingBox(idxs(i),:));
            rect(bbox(2):(bbox(2)+bbox(4)-1), bbox(1):(bbox(1)+bbox(3)-1)) = 1;
            rect_handles{end+1}=rectangle('Position', bbox, 'LineWidth', 2, 'EdgeColor', 'r');
        end
    end
    if issave==1 %保存GIF图
        frame = getframe(gcf);              % 获取当前图形对象的帧
        image_with_border = frame2im(frame);
        if times==1 % figure直接截取会有边框，需要获取边框坐标
            [top_left,bottom_right]=getPosition(240,image_with_border(:,:,1));
        end
        
        % 去除边框并提取原图
        original_image = image_with_border(top_left(2):bottom_right(2),top_left(1):bottom_right(1), :);
        [I,maps] = rgb2ind(original_image,256);
        if(times==first)
            imwrite(I,maps,[pathnums,'.gif'],'DelayTime',0.04,'LoopCount',Inf)
        else
            imwrite(I,maps,[pathnums,'.gif'],'WriteMode','append','DelayTime',0.04)
        end
    end
    pause(0.00001)
end
% 将新图像取反色，得到黑色矩形
rect = ~rect;
%% 画图区
switchs=[1,1,1]; %是否显示图片组
if switchs(1)==1
    figure
    subplot(2,2,1)
    imshow(background)
    title("背景")
    subplot(2,2,2)
    imshow(video)
    title("当前帧")
    subplot(2,2,3)
    imshow(differ)
    title("作差")
    subplot(2,2,4)
    imshow(gray_img)
    title("灰度化")
    hold on
end
if switchs(2)==1
    figure
    subplot(221)
    imshow(bm_img)
    title("二值化")
    subplot(222)
    imshow(open_img)
    title("开运算")
    subplot(223)
    imshow(bw_img_cleaned)
    title("去除小面积")
    subplot(224)
    imshow(close_img)
    title("闭运算")
end
if switchs(3)==1
    figure
    subplot(221)
    imshow(swell_img)
    title("膨胀")
    subplot(222)
    imshow(rect)
    title("矩形化")
    subplot(223)
    imshow(L)
    title("标注矩阵")
    subplot(224)
    imshow(video)
    hold on
    for i = 1:length(idxs)-1
        bbox = round(stats.BoundingBox(idxs(i),:));
        rect(bbox(2):(bbox(2)+bbox(4)-1), bbox(1):(bbox(1)+bbox(3)-1)) = 1;
        rectangle('Position', bbox, 'LineWidth', 2, 'EdgeColor', 'r');
        hold on
    end
    title("标注")
end
%% 获取原图坐标的函数，在保存GIF图要去除figure边框时使用
function [top_left_corner,bottom_right_corner]=getPosition(num,image)
% 寻找原图的四个角的坐标
% 设定边框的灰度值
border_gray_value = num;

% 获取图像的尺寸
[height, width] = size(image);

% 初始化原图的四个角的坐标
top_left_corner = [];
bottom_right_corner = [];

% 寻找左上角的坐标
for y = 1:height
    for x = 1:width
        if image(y, x) ~= border_gray_value
            top_left_corner = [x, y];
            break;
        end
    end
    if ~isempty(top_left_corner)
        break;
    end
end

% 寻找右下角的坐标
for y = height:-1:1
    for x = width:-1:1
        if image(y, x) ~= border_gray_value
            bottom_right_corner = [x, y];
            break;
        end
    end
    if ~isempty(bottom_right_corner)
        break;
    end
end
end