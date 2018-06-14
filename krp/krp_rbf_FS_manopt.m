% takes input nxd matrix.
function [A, violate_ratio] = krp_rbf_FS_manopt(X, thresh, lambda, num_iter, num_subspaces, sigma, verbose) 
    if nargin == 6
        verbose = 0;
    end
    n = size(X,1); 
    z = initialize_z();
    if isempty(sigma)
        sigma = sqrt(median(mypdist2(X, z, 'sqeuclidean')));
    end
    if isempty(thresh)
        thresh=1/n;         
    end
    if isempty(lambda)
        lambda = 10*n;
    end
    kernel = @(a,b) exp(-mypdist2(a,b,'sqeuclidean')/2/sigma^2); 
    get_kxz = @(S,K) diag((S*K)'*K*(S*K));
    K = kernel(X, X); 
    [A,e]=eig(K); [~,idx]=sort(diag(e),'descend'); idx=idx(1:num_subspaces); A=A(:,idx); 
    
    %manifold = symfixedrankYYfactory(n,num_subspaces);          
    manifold = obliquefactory(n,num_subspaces);          
    problem.M = manifold;
    problem.cost = @objective;
    problem.egrad = @grad;
        
    %checkgradient(problem); return;

    opts.maxiter = num_iter;
    opts.verbosity = 3;    
    kxz = get_kxz(A*A', K);    
    [vi, vj] = get_violations(A); violate_init = length(vi);     
    if verbose        
        figure(1);clf(1); plot(kxz);           
        fprintf('before: norm(z)=%f num_violations=%d\n', norm(z), length(vi));
    end
    A = conjugategradient(problem, A, opts);
    %A = trustregions(problem, A, opts);
              
    kxz = get_kxz(A*A', K);
    [vi, vj] = get_violations(A); violate_final = length(vi);
    if verbose        
        figure(2);clf(2); plot(kxz);        
        fprintf('final: norm(z)=%f num_violations=%d\n', norm(z), length(vi));  
    end
    violate_ratio = violate_final/violate_init;
       
    function obj = objective(A)
        P2 = (K*A)*A'; 
        obj = trace(K) + 0.5*(trace((P2 - 2*eye(n))*P2*K));
        [vi, vj] = get_violations(A);
        if ~isempty(vi)
            P3 = (A'*K)*A;        
            %K1 = K(:,vi); K2 = K(:,vj); K12 = K1*K1' - K2*K2';
            uvi = unique(vi); uvj = unique(vj); 
            hvi = hist(vi, uvi); hvj = hist(vj, uvj);
            K12 = fast_KK(uvi, hvi) - fast_KK(uvj, hvj);
            obj = obj + 0.5*lambda*(trace((K12*A)*(P3*A'))+ thresh);
        end        
    end

    function gA = grad(A)
        P1 = K*(K*A); P2 = (K*A)*A'; P3 = A'*(K*A);
        gA = P1*(P3-2*eye(num_subspaces)) + P2*P1;
        [vi,vj] = get_violations(A);
        if ~isempty(vi)
            K1 = K(:,vi)*K(:,vi)'; K2 = K(:,vj)*K(:,vj)'; K12 = K1-K2;
            gA = gA + lambda*(K12*A*P3 + P2*K12*A);
        end        
    end

    function KK = fast_KK(uv, hv)
        KK = (K(:,uv)*diag(hv))*K(:,uv)';
    end

    function [vi, vj] = get_violations(A) 
       kxz = get_kxz(A*A', K);
       vv = kxz*ones(1,n) - ones(n,1)*kxz' + thresh; %xi-xj+1 i<j
       vv = vv - diag(diag(vv));
       vv = triu(vv);
       vv(vv<=0)=0;
       ii = find(vv > 0);
       if any(ii)
           [vi,vj] = ind2sub(size(vv), ii);
       else
           [vi,vj] = deal([]);
       end
    end

   function xx = initialize_z()
        xx = mean(X,1);        
        %xx = krp_linear_manopt(X, thresh, 100, 100, sigma);          
        %[u,s,v]=svd(cov(X));
        %xx=u(:,1)';
    end
end