% Preprocessing for KRP-FS.
% implemented by Anoop Cherian, anoop.cherian@gmail.com
function W = VideoDarwinRepresentation_FS(data, manifold_type, num_subspaces, thresh, num_iter, sigma)
    OneToN = [1:size(data,1)]';    
	if size(data,1) == 1
		fprintf('size of data is one... continuing...\n');
		W{1} = data; W{2} = data;
   		return;
	end

    %Data = cumsum(data,1);
    %Data = Data ./ repmat(OneToN,1,size(Data,2));
	Data = data;
	lambda = []; %1*size(data,1); 
    %thresh = 1/size(Data,1);
	A_fow = krp_rbf_FS_withslack_manopt(getNonLinearity(Data), thresh, lambda, num_iter, num_subspaces, sigma, 100, 0); % X, thresh, lambda, num_iter, num_subspaces, sigma, C, verbose) 
   
    order = 1:size(data,1);
    [~,order] = sort(order,'descend');
    data = data(order,:);
    Data = data;
    %Data = cumsum(data,1);
    %Data = Data ./ repmat(OneToN,1,size(Data,2));
	%%Data = data;
    A_rev = krp_rbf_FS_withslack_manopt(getNonLinearity(Data), thresh, lambda, num_iter, num_subspaces, sigma, 100, 0);
    
    W{1} = A_fow; W{2}= A_rev; 
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
    %Data = sign(Data).*sqrt(abs(Data));    
    %Data = vl_homkermap(Data',2,'kchi2');    
    %Data = vl_homkermap(Data,2,'kchi2');    
end

function x = normalizeL2(x)
    x=x./repmat(sqrt(sum(x.*conj(x),2)),1,size(x,2));
end

function x = normalizeL1(x)
    x = bsxfun(@times, x, 1./sum(abs(x),2));
end
