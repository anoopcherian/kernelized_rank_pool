% an efficient implementation of FS kernel
% uses Nystrom apprximation of the kernel.
function K = compute_FS_kernel4(A1, X1, A2, X2, sigma, traintest, type, dataset, num_subspaces, nystrom_ratio, num_blocks)    
    n1=length(X1); n2=length(X2);
	d = num_subspaces;
	%expproj_linear_kernel = @(a,b,x,y) exp(sum(sum(((b*a')*(x*y')).^2))/2/d);
	myfnorm = @(x,norm_type) sqrt(sum(sum(x.^2)));
	linear_rbf_kernel = @(a,b,x,y,sig) norm(a'*(exp(-mypdist2(x,y,'sqeuclidean')/2/sig^2)*b), 'fro'); 
	%expproj_rbf_kernel = @(a,b,x,y,sig) exp(linear_rbf_kernel(a,b,x,y,sig)/2/d);
	%laplace_rbf_kernel = @(a,b,x,y,sig) exp(-sqrt(1-linear_rbf_kernel(a,b,x,y,sig)^2)/2/d);
    %binomial_rbf_kernel= @(a,b,x,y,sig) (1-linear_rbf_kernel(a,b,x,y,sig)^2)^-0.5;
	trace_kernel = @(a,b,x,y,sig)  trace(a'*((x*y')*b));
	%trace_rbf_kernel = @(a,b,x,y,sig) trace(a'*(exp(-mypdist2(x,y,'sqeuclidean')/2/sig^2)*b));
	%laplace_linear_kernel = @(a,b,x,y,sig,k_xx,k_yy) exp(-0.5*(1-linear_rbf_kernel(a,b,x,y,1)^2/d));
	laplace_linear_kernel = @(a,b,x,y,sig,k_xx,k_yy) exp(-0.5*(1-sqrt(linear_rbf_kernel(a,b,x,y,1)/(d)))); 
	%laplace_linear_kernel = @(a,b,x,y,sig,k_xx,k_yy) exp(linear_rbf_kernel(a,b,x,y,1)/(d)); 

	normalized_expproj_rbf_kernel = @(a, b, x, y, sig, k_xx, k_yy) exp(linear_rbf_kernel(a,b,x,y, sig)/sqrt(k_xx*k_yy)/2/d);
	normalized_laplace_rbf_kernel = @(a, b, x, y, sig, k_xx, k_yy) exp(-sqrt(1-linear_rbf_kernel(a,b,x,y, sig)/(k_xx*k_yy)/d)/2);
	normalized_binomial_rbf_kernel = @(a, b, x, y, sig, k_xx, k_yy) (1-linear_rbf_kernel(a,b,x,y, sig)/(k_xx*k_yy)/d)^(0.5);

	function K_XX = normalizer_KK(aa, xx, ss)
		K_XX = zeros(length(aa),1);
		for t=1:length(aa)
 			%K_XX(t) = expproj_rbf_kernel(aa{t}, aa{t}, xx{t}, xx{t}, ss(t));
 			K_XX(t) = 1; %linear_rbf_kernel(aa{t}, aa{t}, xx{t}, xx{t}, ss(t));
		end
	end

	%kernel_fn = normalized_expproj_rbf_kernel;
	kernel_fn = laplace_linear_kernel;
	%myparpool(3);

	pconst = @(xx) parallel.pool.Constant(xx); % myparpool(4);
    try        
        kernel = parload(['./data/HMDB/' dataset '1_kernel_' traintest '_' type '.mat'], 'kernel');
    catch
        fprintf('computing and caching jhmdb %s kernel...\n', traintest);
        [X1, X2] = preprocess_data(X1, X2, traintest);    
		
		if strcmp(traintest, 'train')
			SIGMA = 1 +0*cellfun(@(xx) (sqrt(mean(mypdist2(xx,mean(xx,1),'sqeuclidean')))), X1); 
			K_XX = normalizer_KK(A1, X1, SIGMA);
   		    SIGMA1 = SIGMA; SIGMA2 = SIGMA; 
			K_XX1 = K_XX; K_XX2 = K_XX;

			% block processing to avoid matlab memory issues. 
			% nystrom
            if nystrom_ratio > 1
                perm = randperm(length(X1)); X1 = X1(perm); A1 = A1(perm); X2 = X2(perm); A2 = A2(perm);
                X1 = X1(1:floor(n1/nystrom_ratio)); n1 = length(X1);A1 = A1(1:n1);
            end
            
			bs1 = max(1,floor(n1/num_blocks)); %floor(n1/3); 
			bs2= max(1,floor(n2/num_blocks)); % use 8 for hmdb
			SIGMA1 = SIGMA(1:n1); K_XX1 = K_XX(1:n1);
            
            if nystrom_ratio > 1
                Kt = compute_kernel_in_blocks(bs1, bs2, kernel_fn, 'test'); Kt = Kt';
                dim = min(size(Kt));			
                W = Kt(1:dim, 1:dim); K = (Kt/W)*Kt';
                K(perm,perm) = K;
            else
               K = compute_kernel_in_blocks(bs1, bs2, kernel_fn, 'test');
            end
			%K = compute_kernel_in_blocks(bs1, bs2, kernel_fn, 'train');
			%K = K + K' - diag(diag(K));
		else			
			SIGMA1 = 1 + 0*cellfun(@(xx) (sqrt(mean(mypdist2(xx,mean(xx,1),'sqeuclidean')))), X1); %SIGMA = mean(SIGMA1)*ones(1,length(SIGMA1));
			SIGMA2 = 1 + 0*cellfun(@(xx) (sqrt(mean(mypdist2(xx,mean(xx,1),'sqeuclidean')))), X2); 
			K_XX1 = normalizer_KK(A1, X1, SIGMA1); %SIGMA1 = SIGMA2; 
			K_XX2 = normalizer_KK(A2, X2, SIGMA2);
			bs1 = max(1,floor(n1/num_blocks)); bs2= max(1,floor(n2/num_blocks)); % use 10 for hmdb
			%K = compute_kernel_in_blocks(X1, X2, A1, A2, SIGMA1, SIGMA2, bs1, bs2, linear_rbf_kernel, 'test')
			K = compute_kernel_in_blocks(bs1, bs2, kernel_fn, 'test');

			%for t=1:n1                                                                                               	
    		%	x1 = X1{t}; a1 = A1{t}; sig1 = SIGMA1(t);
			%	%K(t,:) = arrayfun(@(ii) expproj_linear_kernel(a1, A2{ii}, x1, X2{ii}), 1:n2); 
			%	KK{t} = arrayfun(@(ii)  linear_rbf_kernel(a1, A2{ii}, x1, X2{ii}, (sig1+SIGMA2(ii))/2), 1:n2); 
			%end
			%for t=1:n1, K(t,:) = KK{t}; end			
	    end               
		fprintf('%s -kernel in (%f, %f), mean=%f std=%f\n', traintest, min(K(:)), max(K(:)), mean(K(:)), std(K(:)));                                                                                          
	end

	function K = block_kernel(X1, X2, A1, A2, kernel, sigma1, sigma2, k_xx1, k_xx2, idx1, idx2, traintest)
		n1 = length(X1); n2 = length(X2);
		KK = cell(n1);
		K = zeros(n1, n2);
		oidx = 1:n2; idx = oidx; istest = strcmp(traintest, 'test'); 
		if ~istest
			if any(idx2>=idx1(1)) % if any.
				parfor t=1:n1		
					x1 = X1{t}; a1 = A1{t}; sig1 = sigma1(t); k1 = k_xx1(t);
					idx = oidx(idx2>=idx1(t));
					KK{t} = arrayfun(@(ii)  kernel(a1, A2{ii}, x1, X2{ii}, (sig1+sigma2(ii))/2, k1, k_xx2(ii)), idx); 
				end	
				for t=1:n1,	idx=oidx(idx2>=idx1(t)); K(t,idx) = KK{t}; end
			end
		else
			parfor t=1:n1		
				x1 = X1{t}; a1 = A1{t}; sig1 = sigma1(t); k1 = k_xx1(t);
				KK{t} = arrayfun(@(ii)  kernel(a1, A2{ii}, x1, X2{ii}, (sig1+sigma2(ii))/2, k1, k_xx2(ii)), 1:n2); 
			end	
			for t=1:n1, K(t,:) = KK{t}; end
		end
	end
	
	function K = compute_kernel_in_blocks(bs1, bs2, kernel_fn, traintest)
		n1 = length(X1); n2 = length(X2);
		K = zeros(n1, n2);
		blk1 = [0:bs1:n1-1, n1]; blk2 = [0:bs2:n2-1, n2];
		tot_tme = 0;
		for b1=1:length(blk1)-1
			idx1 = blk1(b1)+1:blk1(b1+1); XX = X1(idx1); AA = A1(idx1);  SS = SIGMA1(idx1); PP = K_XX1(idx1);
			for b2=1:length(blk2)-1
				fprintf('.');
				idx2 = blk2(b2)+1:blk2(b2+1);
				tic;
				myK = block_kernel(XX, X2(idx2), AA, A2(idx2), kernel_fn, SS, SIGMA2(idx2), PP, K_XX2(idx2), idx1, idx2, traintest);
				K(blk1(b1)+1:blk1(b1+1), blk2(b2)+1:blk2(b2+1)) = myK;
				tme = toc; tot_tme = tot_tme + tme;
			end
			fprintf('compute_FS_kernel(): estimated time to finish = %fs : total time so far = %fs\n', tme*length(blk1)*length(blk2), tot_tme);
		end
	end
end

function [X1, X2] = preprocess_data(X1, X2, traintest)
    for ii=1:length(X1)
        X1{ii} = normalize_data(X1{ii}');
    end
    if strcmp(traintest, 'train')        
        X2 = X1;
    else        
        for ii=1:length(X2)
            X2{ii} = normalize_data(X2{ii}');
        end        
    end
end

function x = normalizeL2(x)
    x=x./repmat(sqrt(sum(x.*conj(x),2)),1,size(x,2));
end

function x = normalizeL1(x)
    x = bsxfun(@times, x, 1./sum(abs(x),2));
end

function Data = getNonLinearity(Data)        
    Data = sign(Data).*sqrt(abs(Data));    
	%Data = sqrt(Data);
    %Data = vl_homkermap(Data',2,'kchi2');    
    %Data = vl_homkermap(Data,2,'kchi2');    
end

function Data = normalize_data(Data)
    %OneToN = [1:size(Data,1)]';    
    %Data = cumsum(Data);
    %Data = Data ./ repmat(OneToN,1,size(Data,2));
	%Data = bsxfun(@rdivide, cumsum(Data,1), [1:size(Data,1)]');
	%Data = getNonLinearity(Data);
    %Data = normalizeL2(getNonLinearity(Data));
end

function sigma = get_sigma(X1, X2)
sigma = (sqrt(mean(mypdist2(X1,mean(X1,1),'sqeuclidean'))) + sqrt(mean(mypdist2(X2,mean(X2,1),'sqeuclidean'))))/2;
end

