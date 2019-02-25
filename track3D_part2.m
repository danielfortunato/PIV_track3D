function [objects, cam2toW] = track3D_part2( imgseq1, imgseq2,   cam_params)
	%% 1. Get our data ready

	% load
	[~, imgsrt1, imgsd1]=loader(imgseq1, cam_params);
	[film_length, imgsrt2, imgsd2]=loader(imgseq2, cam_params);
	fprintf("done loading\n");
	% and do correspondence
	[R21, T21] = get_RT21(film_length, imgsrt1, imgsrt2, imgsd1, imgsd2, cam_params);
	cam2toW.R = R21;
	cam2toW.T = T21;
    fprintf("done RT\n");

	%% 2. Now box objects moving
	[image_objects, image_pcs] = extract_objects(film_length, imgsrt1, imgsrt2, imgsd1, imgsd2, R21, T21, cam_params);
	fprintf("done extracting\n");

	%% 3. Identify which objects are the same, creating a path between boxes
	% gettin maximum volume and maximum norm to normalize the costs 0-1
	[maxvol, maxnorm] = get_maxes(film_length, imgsd1, cam_params);
	objects = get_path(film_length, image_objects, image_pcs, maxvol, maxnorm);
	fprintf("done tracking\n");

end

% -------------------------------------------------------------------------------------- %
% ------------------------------------------1------------------------------------------- %
% -------------------------------------------------------------------------------------- %

% Input - cam_params struct, imgseq1 struct of image names
% Output - fl film lenght, imgsrt are rgb rotated and translated to world, 
% imgsd depth images 

function [fl, imgsrt, imgsd] = loader( imgseq, cam_params)
    % Make use of the arguments

    % empty images rgb, rgb R & T, depth
    imgs=zeros(480,640,3,length(imgseq));
    imgsrt=zeros(480,640,3,length(imgseq));
    imgsd=zeros(480,640,length(imgseq));

    % film length
    fl = length(imgseq);
    
    % Load information and compute digital RGB camera
    for i=1:length(imgseq)
        d=dir(imgseq(i).rgb);
        dd=dir(imgseq(i).depth);
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

% Input - 
% Output - 


function [R21, T21] = get_RT21(film_length, imgsrt1, imgsrt2, imgd1, imgd2, cam_params)
    %% Images selected
    samples=fix(0.10*film_length);
    img_index=fix(rand(samples,1)*film_length)+1;
    
    for i=1:length(img_index)
        dp1(:,:,i)=imgd1(:,:,img_index(i));
        dp2(:,:,i)=imgd2(:,:,img_index(i));
        rgbd1(:,:,:,i)=imgsrt1(:,:,:,img_index(i));
        rgbd2(:,:,:,i)=imgsrt2(:,:,:,img_index(i));        
    end
    
	%% Compute 3d in camera frame
    [v, u]=ind2sub([480 640],(1:480*640));
    for i=1:length(img_index)
        % cam 1
        Z = dp1(:,:,i);
        Z = Z(:)';
        P=inv(cam_params.Kdepth)*[Z.*u ;Z.*v;Z];
        xyz1(:,:,i) = P';
        % cam 2
        Z = dp2(:,:,i);
        Z = Z(:)';
        P=inv(cam_params.Kdepth)*[Z.*u ;Z.*v;Z];
        xyz2(:,:,i) = P';
    end
    %% Perform SIFT for all images 
    P1=[];
    P2=[];

    for i=1:length(img_index)
        I1=single(rgb2gray(uint8(rgbd1(:,:,:,i))));
        I2=single(rgb2gray(uint8(rgbd2(:,:,:,i))));
        [f1,d1_]=vl_sift(I1);
        [f2,d2_]=vl_sift(I2);
        [matches, ~] = vl_ubcmatch(d1_, d2_, 3);
        y1=round(f1(2,matches(1,:)));
        x1=round(f1(1,matches(1,:)));
        y2=round(f2(2,matches(2,:)));
        x2=round(f2(1,matches(2,:)));
        ind1=sub2ind(size(dp2(:,:,i)),y1,x1);
        ind2=sub2ind(size(dp2(:,:,i)),y2,x2);
        p1=xyz1(ind1,:,i);
        p2=xyz2(ind2,:,i);
        final_inds=find((p1(:,3).*p2(:,3))>0);
        p1=p1(final_inds,:);p2=p2(final_inds,:);
        P1=[P1; p1];
        P2=[P2; p2];
    end
    
    %% Ransac
    n_it=1000;
    threshold=0.4;
    num_in=[];
    % Generate random numbers
    n_points=fix(rand(4*n_it,1)*length(P1))+1;

    for i=1:n_it-3
        % Choose random points, we need at least 4 points for the procrustes
        rand_points=n_points(4*i:4*i+3);
        % Get 3D points correspondent to theses indexes

        X1=P1(rand_points,:);
        X2=P2(rand_points,:);

        % Compute procrustes
        [~, ~, transform]=procrustes(X1,X2, 'scaling', false, 'reflection', false);

        % Compute points from image 2 projected to image 1 using the
        % transformation from procrustes

        xyz21= P2 * transform.T + + ones(length(P2),1)*transform.c(1,:);
        error=sqrt(sum((P1-xyz21).^2,2)); % Compute the error for all points

        inds(i).in=find(error<threshold);

        % Add the number of inliers to the inliers vector

        num_in=[num_in length(inds(i).in)];
    end
    
    [~, ind]=max(num_in);
    xyz1f=P1(inds(ind).in,:);
    xyz2f=P2(inds(ind).in,:);
    [~, ~, transform]=procrustes(xyz1f, xyz2f, 'scaling', false, 'reflection', false);
    R21=transform.T;
    T21=transform.c(1,:);

end


% -------------------------------------------------------------------------------------- %
% ------------------------------------------2------------------------------------------- %
% -------------------------------------------------------------------------------------- %

function [image_objects, image_pcs] = extract_objects(film_length, imgsrt1, imgsrt2, imgsd1, imgsd2, R21, T21,cam_params)


    % find the background with median ON DEPTH
    bgdepth1=median(imgsd1,3);
    bgdepth2=median(imgsd2,3);

    % pre alocate structs
    image_objects(film_length) = struct();
    image_pcs(film_length) = struct();
    image_objects2(film_length) = struct();
    image_pcs2(film_length) = struct();

    % use the depth image to check objects moving (the best)
    for i=1:film_length

        % FIRST CAMERA 
        % subtract background
        imdiff=abs(imgsd1(:,:,i)-bgdepth1)>0.20;
        % filter image, maybe another filter to get better objects?
        imdifil=imopen(imdiff,strel('disk',6));
            
        % check with gradients between overlapping objects
        [Gmag, ~] = imgradient(imgsd1(:,:,i));
        lin_indexes = Gmag > 1;
        imdifil(lin_indexes) = 0;
        
        % label every object
        [L, num]=bwlabel(imdifil);
                
        % then box it
        object_num = 1;
        for j = 1:num
            % check indexes for each label and compute 2d extremes
            [rows, columns] = find(L == j);
            pixel_list = [rows, columns];
            % check if area is at least 3000 pixels ~5%
            if length(pixel_list) < 3000
                continue
            end        
            
            % compute point cloud and store it for further use
            R11 = [1,0,0; 0,1,0; 0,0,1];
            T11 = [0 0 0];
            pc = get_object_pc(pixel_list, imgsrt1(:,:,:,i), imgsd1(:,:,i), R11, T11, cam_params);
            % get values from point cloud
            [xmin, xmax, ymin, ymax, zmin, zmax]=getboundingbox(pc);
            
            image_pcs(i).object{object_num} = pc;
            
            % set up the final object struct
            image_objects(i).object(object_num).X = [xmin, xmin, xmin, xmin, xmax, xmax, xmax, xmax];
            image_objects(i).object(object_num).Y = [ymax, ymax, ymin, ymin, ymax, ymax, ymin, ymin];
            image_objects(i).object(object_num).Z = [zmax, zmin, zmin, zmax, zmax, zmin, zmin, zmax];
            image_objects(i).object(object_num).frames_tracked = i;
            
            % next object
            object_num = object_num +1;
        end
        curr_objects = object_num;


        % SECOND CAMERA
        % subtract background
        imdiff=abs(imgsd2(:,:,i)-bgdepth2)>0.20;
        % filter image, maybe another filter to get better objects?
        imdifil=imopen(imdiff,strel('disk',6));
            
        % check with gradients between overlapping objects
        [Gmag, ~] = imgradient(imgsd2(:,:,i));
        lin_indexes = Gmag > 1;
        imdifil(lin_indexes) = 0;
        
        % label every object
        [L, num]=bwlabel(imdifil);
                
        % then box it
        object_num = 1;
        for j = 1:num
            % check indexes for each label and compute 2d extremes
            [rows, columns] = find(L == j);
            pixel_list = [rows, columns];
            % check if area is at least 3000 pixels ~5%
            if length(pixel_list) < 3000
                continue
            end        
            
            % compute point cloud in world frame
            pc = get_object_pc(pixel_list, imgsrt2(:,:,:,i), imgsd2(:,:,i), R21, T21, cam_params);

            % get values from point cloud
            [xmin, xmax, ymin, ymax, zmin, zmax]=getboundingbox(pc);
            
            image_pcs2(i).object{object_num} = pc;
            
            % set up the final object struct
            image_objects2(i).object(object_num).X = [xmin, xmin, xmin, xmin, xmax, xmax, xmax, xmax];
            image_objects2(i).object(object_num).Y = [ymax, ymax, ymin, ymin, ymax, ymax, ymin, ymin];
            image_objects2(i).object(object_num).Z = [zmax, zmin, zmin, zmax, zmax, zmin, zmin, zmax];
            image_objects2(i).object(object_num).frames_tracked = i;
            
            % next object
            object_num = object_num +1;
        end


        % MATCH OBJECTS OF CAMERAS
        object_num = curr_objects;
        Pconst = 1;
        treshold = 0.80; %in meters

        % compute a cost table between the two images
        costtable = ones(length(image_objects(i)),length(image_objects2(i)))*(treshold + 1);    
        for n = 1:length(image_objects(i).object)
            for m = 1:length(image_objects2(i).object)
                % proximity cost is distance:
                costtable(n,m) = Pconst * cost_proximity(image_objects(i).object(n), image_objects2(i).object(m));
            end
        end
        % get correspondence of the second camera to the first camera
        [index_object] = greedy(costtable, length(image_objects(i).object), length(image_objects2(i).object), treshold);

        % add objects of camera two that are non existant in camera one
        for m = 1:length(image_objects2(i).object)
            % if there wasn't a match, it's an object only on camera 2
            if index_object(m) < 0
                image_objects(i).object(object_num) = image_objects2(i).object(m);
                image_pcs(i).object{object_num} = image_pcs2(i).object{m};
                object_num = object_num + 1;
            % if there was, join both informations
            else
                % merge with 2cm precision
                pc_merged = pcmerge(image_pcs(i).object{index_object(m)}, image_pcs2(i).object{m}, 0.02);
                image_pcs(i).object{object_num} = pc_merged;
                [xmin, xmax, ymin, ymax, zmin, zmax]=getboundingbox(pc_merged);
                % change the box, the frame is the same
                image_objects(i).object(object_num).X = [xmin, xmin, xmin, xmin, xmax, xmax, xmax, xmax];
                image_objects(i).object(object_num).Y = [ymax, ymax, ymin, ymin, ymax, ymax, ymin, ymin];
                image_objects(i).object(object_num).Z = [zmax, zmin, zmin, zmax, zmax, zmin, zmin, zmax];
            end
        end

    end

end


% -------------------------------------------------------------------------------------- %

% Input - pixel_list of a singular object, imgrgb of that object, imgdepth of that object
% rotation and translation to world, camera parameteres
% Output - pc point cloud of a singular object in world frame

function [pc]=get_object_pc(pixel_list, imgrgb,imgdepth, R21, T21, cam_params)

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

	% compute xyz
	Z=double(depth_index(:)');
	u=pixel_list(:,2)';
	v=pixel_list(:,1)';
	P=inv(cam_params.Kdepth)*[Z.*u ;Z.*v;Z];
	% rotate and translate, T21 horizontal
	xyz = (P')*R21 + repmat(T21, length(P'), 1);
	pc=pointCloud(xyz,'color',uint8(rgb_index));

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
    
    cost = norm(c1-c2);
   
end

% -------------------------------------------------------------------------------------- %

% Input - object1 and object2 boxes
% Output - cost of distance between centroids

function [cost] = cost_volume(object1, object2)
  
    % objects volumes
    V1 = (max(object1.X)-min(object1.X))*(max(object1.Y)-min(object1.Y))*(max(object1.Z)-min(object1.Z));
    V2 = (max(object2.X)-min(object2.X))*(max(object2.Y)-min(object2.Y))*(max(object2.Z)-min(object2.Z));
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