function [objects] = track3D_part1( imgseq1,   cam_params)
	%% 1. Get our data ready

	[film_length, imgsrt, imgsd]=loader(imgseq1, cam_params);
	fprintf("done loading\n");

	%% 2. Now box objects moving
	% we extract object boxes and points clouds to further use

	[image_objects, image_pcs] = extract_objects(film_length, imgsrt, imgsd, cam_params);
	fprintf("done extracting\n");

	%% 3. Identify which objects are the same, creating a path between boxes
	% connect boxes paths using proximity of boxes, volume and colour

	% gettin maximum volume and maximum norm to normalize the costs 0-1
	[maxvol, maxnorm] = get_maxes(film_length, imgsd, cam_params);
	objects = get_path(film_length, image_objects, image_pcs, maxvol, maxnorm);
	fprintf("done tracking\n");

end


% -------------------------------------------------------------------------------------- %
% ------------------------------------------1------------------------------------------- %
% -------------------------------------------------------------------------------------- %

% Input - cam_params struct, imgseq1 struct of image names
% Output - fl film lenght, imgsrt are rgb rotated and translated to world, 
% imgsd depth images 

function [fl, imgsrt, imgsd] = loader( imgseq1, cam_params)
	% Make use of the arguments

	% empty images rgb, rgb R & T, depth
	imgs=zeros(480,640,3,length(imgseq1));
	imgsrt=zeros(480,640,3,length(imgseq1));
	imgsd=zeros(480,640,length(imgseq1));

	% film length
	fl = length(imgseq1);

	% Load information and compute digital RGB camera
	for i=1:length(imgseq1)
	    d=dir(imgseq1(i).rgb);
	    dd=dir(imgseq1(i).depth);
	    % load the information
	    imgs(:,:,:,i)=imread(fullfile(d.folder,d.name));
	    load(fullfile(dd.folder,dd.name));
	    Z=double(depth_array(:)')/1000;
	    
	    % compute correspondence
	    [v, u]=ind2sub([480 640],(1:480*640));
	    P=inv(cam_params.Kdepth)*[Z.*u ;Z.*v;Z];
	    niu=cam_params.Krgb*[cam_params.R cam_params.T]*[P;ones(1,640*480)];
	    u2=round(niu(1,:)./niu(3,:));
	    v2=round(niu(2,:)./niu(3,:));
	    
	    % and compute new image
	    im=imread(fullfile(d.folder,d.name));
	    im2=zeros(640*480,3);
		indsclean=find((u2>=1)&(u2<=641)&(v2>=1)&(v2<=480));
		indscolor=sub2ind([480 640],v2(indsclean),u2(indsclean));
		im1aux=reshape(im,[640*480 3]);
		im2(indsclean,:)=im1aux(indscolor,:);
	    aux=uint8(reshape(im2,[480,640,3]));
	    imgsrt(:,:,:,i)=aux;
	    
	    % finally save our depth array for further use
	    imgsd(:,:,i)=double(depth_array)/1000;
	    
	end
end


% -------------------------------------------------------------------------------------- %
% ------------------------------------------2------------------------------------------- %
% -------------------------------------------------------------------------------------- %

% Input - film_length, imgsrt rgb images on depth coordinates, imgsd depth images and
% cam_params camera parameters
% Output - image_objects struct with a struct of type object (like final struct) for each
% object for each image, image_pcs a struct of point clouds for each object in each image.

function [image_objects, image_pcs] = extract_objects(film_length, imgsrt, imgsd, cam_params)

	% find the background with median ON DEPTH
	bgdepth=median(imgsd,3);

	% pre alocate structs
	image_objects(film_length) = struct();
	image_pcs(film_length) = struct();

	% use the depth image to check objects moving (the best)
	for i=1:film_length
	    % subtract background
	    imdiff=abs(imgsd(:,:,i)-bgdepth)>0.20;
	    % filter image, maybe another filter to get better objects?
	    imgdiffiltered=imopen(imdiff,strel('disk',6));
	        
	    % check with gradients between overlapping objects
	    [Gmag, ~] = imgradient(imgsd(:,:,i));
	    lin_indexes = Gmag > 0.5;
	    imgdiffiltered(lin_indexes) = 0;
	    
	    % label every object
	    [L, num]=bwlabel(imgdiffiltered);
	            
	    % then box it
	    p = 1;
	    for j = 1:num
	        % check indexes for each label and compute 2d extremes
	        [rows, columns] = find(L == j);
	        pixel_list = [rows, columns];
	        % check if area is at least 3000 pixels ~5%
	        if length(pixel_list) < 3000
	            continue
	        end        
	        
	        % compute point cloud and store it for further use
	        pc = get_object_pc(pixel_list, imgsrt(:,:,:,i), imgsd(:,:,i), cam_params);
	        % get values from point cloud
	        [xmin, xmax, ymin, ymax, zmin, zmax]=getboundingbox(pc);
	        
	        image_pcs(i).object{p} = pc;
	        
	        % set up the final object struct
	        image_objects(i).object(p).X = [xmin, xmin, xmin, xmin, xmax, xmax, xmax, xmax];
	        image_objects(i).object(p).Y = [ymax, ymax, ymin, ymin, ymax, ymax, ymin, ymin];
	        image_objects(i).object(p).Z = [zmax, zmin, zmin, zmax, zmax, zmin, zmin, zmax];
	        image_objects(i).object(p).frames_tracked = i;
	        
	        % next object
	        p = p +1;
	    end
	end

end


% -------------------------------------------------------------------------------------- %

% Input - pixel_list of a singular object, imgrgb of that object, imgdepth of that object
% and camera parameteres
% Output - pc point cloud of a singular object

function [pc]=get_object_pc(pixel_list, imgrgb,imgdepth, cam_params)

	rgb_index=zeros(length(pixel_list),3);
	depth_index=zeros(length(pixel_list),1);

	% Get the pixel values from the rgb image 
	for i=1:length(pixel_list)
	    rgb_index(i,:)=imgrgb(pixel_list(i,1),pixel_list(i,2),:);
	end

	% Get the pixel values from the depth image 

	for i=1:length(pixel_list)
	    depth_index(i)=imgdepth(pixel_list(i,1),pixel_list(i,2));
	end

	Z=double(depth_index(:)');
	u=pixel_list(:,2)';
	v=pixel_list(:,1)';
	P=inv(cam_params.Kdepth)*[Z.*u ;Z.*v;Z];
	pc=pointCloud(P','color',uint8(rgb_index));

end


% -------------------------------------------------------------------------------------- %

% Input - pc an object point cloud
% Output - limits of an object box

function [xmin, xmax, ymin, ymax, zmin, zmax]=getboundingbox(pc)

	xmin=min(pc.Location(:,1));
	xmax=max(pc.Location(:,1));
	ymin=min(pc.Location(:,2));
	ymax=max(pc.Location(:,2));
	zmin=min(pc.Location(:,3));
	zmax=max(pc.Location(:,3));

end


% -------------------------------------------------------------------------------------- %
% ------------------------------------------3------------------------------------------- %
% -------------------------------------------------------------------------------------- %

% Input - film_length, imgsd image depth, camera parameters
% Output - maximum volume (in cubic meters) and maximum distance (in meters)

function [maxvol, maxnorm]=get_maxes(film_length, imgsd, cam_params)
	% create the vars
	maxvol = 0;
	maxnorm = 0;

	for i = 1:film_length
	    % create a point cloud, to get x,y in meters
	    depth_array = imgsd(:,:,i);
	    Z=double(depth_array(:)');
	    [v, u]=ind2sub([480 640],(1:480*640));
	    P=inv(cam_params.Kdepth)*[Z.*u ;Z.*v;Z];
	    
	    % extract maximum and minimums
	    posmin = min(P');
	    posmax = max(P');
	    
	    % create new vol and norm
	    newvol = abs(posmax(1)-posmin(1))*abs(posmax(2)-posmin(2))*abs(posmax(3)-posmin(3));
	    newnorm = norm([posmax(1), posmax(2), posmax(3)]- [posmin(1), posmin(2), posmin(3)]);
	    
	    if maxvol < newvol
	        maxvol = newvol;
	    end
	    if maxnorm < newnorm
	        maxnorm = newnorm;
	    end
	end
end


% -------------------------------------------------------------------------------------- %

% Input - film_length, struct of object boxes per image, point cloud of objects per image
% maximum volume and maximum norm.
% Output - the final objects structure, for each object box points and frames tracked

function [objects] = get_path(film_length, image_objects, image_pcs, maxvol, maxnorm)

	% change this scalers to get better cost (K <= 1)
	Pconst = 1;
	Vconst = 1;
	Cconst = 1;
	treshold = 1; %max cost is 3

	% and use them to normalize
	Pconst = Pconst*(1/maxnorm);
	Vconst = Vconst*(1/maxvol);

	% allocate memory before
	costs(film_length) = struct();
	objects(length(image_objects(1).object)) = struct();

	% to keep tracking of the objects
	object_index = zeros(1,length(image_objects(1).object));
	total_objs = 0;

	% first "wave" of objects, no matching yet
	for i = 1:length(image_objects(1).object)
	    object_index(i) = total_objs+1;
	    total_objs = total_objs+1;
	    objects(object_index(i)).X = [image_objects(1).object(object_index(i)).X];
	    objects(object_index(i)).Y = [image_objects(1).object(object_index(i)).Y];
	    objects(object_index(i)).Z = [image_objects(1).object(object_index(i)).Z];
	    objects(object_index(i)).frames_tracked = [image_objects(1).object(object_index(i)).frames_tracked];
	end

	% for each image calculate cost table
	for i=1:(film_length-1)
	    % start all with an impossible cost
	    costs(i).table = ones(length(image_objects(i)),length(image_objects(i+1)))*(treshold+1);
	    
	    % for each pair define a cost
	    for n = 1:length(image_objects(i).object)
	        for m = 1:length(image_objects(i+1).object)
	            % proximity cost is distance:
	            costs(i).table(n,m) = Pconst * cost_proximity(image_objects(i).object(n), image_objects(i+1).object(m));
	            % we can have a volume cost
	            costs(i).table(n,m) = costs(i).table(n,m) + Vconst * cost_volume(image_objects(i).object(n), image_objects(i+1).object(m));
	            
	            % the colour cost should be done with hue and/or saturation
	            costs(i).table(n,m) = costs(i).table(n,m) + Cconst * cost_colour(image_pcs(i).object{n}, image_pcs(i+1).object{m});
	        end
	    end
	      
	    % Assign with greedy algorithm
	    [index_object] = greedy(costs(i).table, length(image_objects(i).object),length(image_objects(i+1).object), treshold);
	        
	    % and make the final object struct
		object_index_aux = zeros(1,length(image_objects(i+1).object));
	    for m = 1:length(image_objects(i+1).object)
	    	% if there was a match
			if index_object(m) > 0
				% track object
	            objects(object_index(index_object(m))).X = [objects(object_index(index_object(m))).X;image_objects(i+1).object(m).X];
	            objects(object_index(index_object(m))).Y = [objects(object_index(index_object(m))).Y;image_objects(i+1).object(m).Y];
	            objects(object_index(index_object(m))).Z = [objects(object_index(index_object(m))).Z;image_objects(i+1).object(m).Z];
	            objects(object_index(index_object(m))).frames_tracked = [objects(object_index(index_object(m))).frames_tracked, image_objects(i+1).object(m).frames_tracked];
	            % register for next iteration
	            object_index_aux(m) = object_index(index_object(m));
			else
				% create new object
	    		total_objs = total_objs+1;
	            objects(total_objs).X = [image_objects(i+1).object(m).X];
	            objects(total_objs).Y = [image_objects(i+1).object(m).Y];
	            objects(total_objs).Z = [image_objects(i+1).object(m).Z];
	            objects(total_objs).frames_tracked = [image_objects(i+1).object(m).frames_tracked];
	            % register for next iteration
				object_index_aux(m) = total_objs;
			end 
	    end
	    % this is the list of objects existant in the next frame
	    object_index = object_index_aux;

	end
end

% -------------------------------------------------------------------------------------- %

% Input - object1 and object2 boxes
% Output - cost of distance between centroids

function [cost] = cost_proximity(object1, object2)
  
    % objects centroids
    c1 = [mean(object1.X), mean(object1.Y), mean(object1.Z)];
    c2 = [mean(object2.X), mean(object2.Y), mean(object2.Z)];
    % distance
    cost = norm(c1-c2);
   
end


% -------------------------------------------------------------------------------------- %

% Input - object1 and object2 boxes
% Output - cost of distance between centroids

function [cost] = cost_volume(object1, object2)
  
    % objects centroids
    V1 = (max(object1.X)-min(object1.X))*(max(object1.Y)-min(object1.Y))*(max(object1.Z)-min(object1.Z));
    V2 = (max(object2.X)-min(object2.X))*(max(object2.Y)-min(object2.Y))*(max(object2.Z)-min(object2.Z));
    % volume dif
    cost = abs(V1-V2);
   
end


% -------------------------------------------------------------------------------------- %


% Input - object1 and object2 point clouds
% Output - cost of colour between point clouds


function [cost] = cost_colour(pc1, pc2)
    % a cost of colour is histogram diference! let's try it
  
    % swap from rgb to hsv
    % this is a cost with the colours having two object rgb pixels.
    hsv1 = rgb2hsv(double(pc1.Color)./255);
    hsv2 = rgb2hsv(double(pc2.Color)./255);
 
    % if saturation is too little we must ignore hue and do with saturation
    linind1 = find(hsv1(:,2)<.05);
    linind2 = find(hsv2(:,2)<.05);
    % the 50000 was random ~16%, img of size 480*640 =~ 300k
    if length(linind1) > 50000 || length(linind2) > 50000
    	% use the saturation
        h1 = imhist(hsv1(:,2),256);
        h2 = imhist(hsv2(:,2),256);
        %normalize
        h1=h1/sum(h1);
        h2=h2/sum(h2);
    else
        % use the hue
        h1 = imhist(hsv1(:,1),256);
        h2 = imhist(hsv2(:,1),256);
        %normalize
        h1=h1/sum(h1);
        h2=h2/sum(h2);
    end

    % returns a cost between 0 and 1
    cost = pdist2(h1',h2');
end


% -------------------------------------------------------------------------------------- %

% Input - cost table between objects total of num_n and num_m, treshold to pick 
% Output - index of object n matching the current object m.

function [index_object] = greedy(cost_table, num_n, num_m, treshold)
	% allocate memory
	match_object = ones(1,num_n)*(treshold + 1);
	index_cost = ones(1,num_n)*(-1);
    index_object = ones(1,num_m)*(-1);

    % for the minimum in each origin object
    for n = 1:num_n
        for m = 1:num_m
            if match_object(n) > cost_table(n,m) && cost_table(n,m) < treshold
            	% pick lowest cost in a row
                match_object(n) = cost_table(n,m);
                index_cost(n) = m;
            end
        end
    end

    % check which of the columns is the lowest value
    for m = 1:num_m
        index = find(index_cost == m);
        % if there is only one value, it is it
        if length(index) == 1
            index_object(m) = index;
        else
            % if there are multiple, select the lowest
            for p = 1:(length(index)-1)
                if match_object(index(p)) < match_object(index(p+1))
                    match_object(index(p+1)) = match_object(index(p));
                    index_object(m) = index(p+1);
                else
                    index_object(m) = index(p);
                end  
            end
        end
    end
end
