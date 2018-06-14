% original Video Darwin Code. Copyright owned by B. Fernando. 
function W = VideoDarwinRepresentation(data,CVAL)
    OneToN = [1:size(data,1)]';    
    Data = cumsum(data);
    Data = Data ./ repmat(OneToN,1,size(Data,2));
    %Data = data;
    W_fow = liblinearsvr(getNonLinearity(Data),CVAL,2); clear Data;         
    order = 1:size(data,1);
    [~,order] = sort(order,'descend');
    data = data(order,:);
    %Data = data;
    Data = cumsum(data);
    Data = Data ./ repmat(OneToN,1,size(Data,2));
    W_rev = liblinearsvr(getNonLinearity(Data),CVAL,2);     
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
% 	for i = 1 : size(X,1)
% 		if norm(X(i,:)) ~= 0
% 			X(i,:) = X(i,:) ./ norm(X(i,:));
% 		end
%     end
    x=x./repmat(sqrt(sum(x.*conj(x),2)),1,size(x,2));
end

function x = normalizeL1(x)
% 	for i = 1 : size(X,1)
% 		if norm(X(i,:)) ~= 0
% 			X(i,:) = X(i,:) ./ norm(X(i,:));
% 		end
%     end
    %x=x./repmat(sqrt(sum(x.*conj(x),2)),1,size(x,2));
    x = bsxfun(@times, x, 1./sum(abs(x),2));
end