% Preprocessing for KRP variants (except KRP-FS).
% implemented by Anoop Cherian, anoop.cherian@gmail.com
function W = VideoDarwinRepresentation_KRP(data, type, CVAL, thresh)
	%max_frames = 100; 
	%if size(data,1)>max_frames, data = data(sort(randperm(size(data,1),max_frames)),:); end % to speed things up!
    OneToN = [1:size(data,1)]';    
	if size(data,1) == 1
		fprintf('VideoDarwinRepresentation_KRP: only one data sample... returning...\n');
		W = [data, data]; return;
	end

    Data = cumsum(data);
    Data = Data ./ repmat(OneToN,1,size(Data,2));
	if nargin == 3
		thresh = 0.01;
    end
    %Data = data;
	lambda = size(data,1);
	switch type
		case 'linear'
	    	W_fow = liblinearsvr(getNonLinearity(Data),CVAL,2); clear Data; 

		case 'RBF-XRecon-WithSlack'
	    	W_fow = krp_rbf_Xreconstruct_withslack_manopt(getNonLinearity(Data), thresh, lambda, 1, 100, []); %X, thresh, lambda, num_iter, sigma)
		
		case 'RBF-Mult-WithSlack'
		   W_fow = krp_rbf_mult_withslack_manopt(getNonLinearity(Data), thresh, lambda, 100, 1, []); % X, thresh, lambda, num_iter, C, sigma)

		case 'RBF-WithSlack'
		    W_fow = krp_rbf_withslack_manopt(getNonLinearity(Data), thresh, lambda, 100, 1, []); %(X, thresh, lambda, num_iter, C, sigma) 
	end
    
    %W_fow = krp_linear_manopt(Data, 0.01, size(Data,1), 100, [], 0);
    order = 1:size(data,1);
    [~,order] = sort(order,'descend');
    data = data(order,:);
    %Data = data;
    Data = cumsum(data);
    Data = Data ./ repmat(OneToN,1,size(Data,2));
	switch type
		case 'linear'
	    	W_rev = liblinearsvr(getNonLinearity(Data),CVAL,2); 

		case 'RBF-XRecon-WithSlack'
	   		W_rev = krp_rbf_Xreconstruct_withslack_manopt(getNonLinearity(Data), thresh, lambda, 1, 100, []);

		case 'RBF-Mult-WithSlack'
	    	W_rev = krp_rbf_mult_withslack_manopt(getNonLinearity(Data), thresh, lambda, 100, 1, []);

		case 'RBF-WithSlack'
		    W_rev = krp_rbf_withslack_manopt(getNonLinearity(Data), thresh, lambda, 100, 1, []); %(X, thresh, lambda, num_iter, C, sigma) 
	end
    %W_rev = krp_linear_manopt(Data, 0.01, size(Data,1), 100, [], 0);
    W = [W_fow, W_rev]; 
    %W = W_fow;
end

function w = liblinearsvr(Data,C,normD)        
    if normD == 2
        Data = normalizeL2(Data);
    end    
    if normD == 1
        Data = normalizeL1(Data);
    end    
    Data = [Data,ones(size(Data,1),1)]; % append one more dimension.
    Labels = [1:size(Data,1)]';    
    %Labels = cumsum(rand(size(Data,1),1)); Labels = Labels/max(Labels);
    model = train(double(Labels), sparse(double(Data)),sprintf('-c %1.6f -s 11 -n 8 -q',C) );    
    w = model.w';    
end

function Data = getNonLinearity(Data)        
    Data = sign(Data).*sqrt(abs(Data));    
    %Data = vl_homkermap(Data',2,'kchi2');    
    %Data = vl_homkermap(Data,2,'kchi2');    
end

function x = normalizeL2(x)

    x=x./repmat(sqrt(sum(x.*conj(x),2)),1,size(x,2));
end

function x = normalizeL1(x)
    x = bsxfun(@times, x, 1./sum(abs(x),2));
end
