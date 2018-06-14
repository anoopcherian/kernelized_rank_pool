
% Implementation of KRP-FS with Slack variables. 
% Input: 
% X (required): an nxd matrix of features, n is the number of frames, d is the dimension of each feature.
% thresh (default 1/n): is the ranking threshold (eta in the paper).
% lambda (default 10n): : is the penalty constant on the order constraints.
% num_iter (default 100): is the number of iterations for the Riemannian gradient.
% num_subspaces (default 5): the number of subspaces to use in KRP. 
% sigma (default will be computed from data): bandwidth of the kernel matrix to be constructed.
% C (default 1): the penalty on the slack.
% verbose = 1, if you want to print some intermediate results or plots.
% 
% implemented by Anoop Cherian, anoop.cherian@gmail.com
function A = krp_rbf_FS_withslack_manopt(X, thresh, lambda, num_iter, num_subspaces, sigma, C, verbose)     
    n = size(X,1);  
    z = initialize_z();
    
    if isempty(thresh) || nargin <= 1
        thresh=1/n;         
    end
    if isempty(lambda) || nargin <= 2 
        lambda = 10*n;
    end    
    if isempty(num_iter) || nargin <= 3
        num_iter = 100;
    end
    if isempty(num_subspaces) || nargin <= 4
        num_subspaces = 5;
    end     
    if isempty(sigma) || nargin <= 5
        sigma = sqrt(mean(mypdist2(X, z, 'sqeuclidean')));
    end
    if isempty(C) || nargin <= 6
        C = 1;
    end
    if nargin <= 7
        verbose = 0;
    end
    
    % define the kenrel function. RBF in this case.
    kernel = @(a,b) exp(-mypdist2(a,b,'sqeuclidean')/2/sigma^2); 
    
    function kxz = get_kxz(SS,KK) 
        PP = SS*KK;
        kxz=diag(PP'*(KK*(PP)));        
    end
    
    K = kernel(X, X)+eye(n)*1e-9;  %K = K - sum(K,2)*ones(1,n) - ones(n,1)*sum(K,1) - sum(sum(K)); % center the kernel.    
    [A,e]=eig(K); [~,idx]=sort(diag(e),'descend'); idx=idx(1:num_subspaces); A=A(:,idx); 
    xi = mean(mean(X))*ones(1,n*(n-1)/2)/100;
    trK = trace(K);
    
    % cache valid indices in the violation matrix
    idx = triu(ones(n,n)) - diag(ones(n,1));
    idx = find(idx);
          
    manifold.A = grassmanngeneralizedfactory(n, num_subspaces, K); % generalized grassmann factory.
    manifold.S = euclideanfactory(1, n*(n-1)/2);
    problem.M = productmanifold(manifold);
    problem.cost = @objective;
    problem.egrad = @grad;
        
    init.A = A; init.S = xi;    

    opts.maxiter = num_iter;
    opts.verbosity = 0;
    
    if verbose
     kxz = get_kxz(A*A', K);    
     [vi, vj] = get_violations(A, xi); violate_init = length(vi);     
         figure(1);clf(1); plot(kxz); figure(3); plot(xi); title('xi');          
         fprintf('before: norm(z)=%f num_violations=%d\n', norm(z), length(vi));
    end   
    out = conjugategradient(problem, init, opts);
    A = out.A; xi = out.S;
    
    if verbose          
        kxz = get_kxz(A*A', K);
        [vi, vj] = get_violations(A, xi); violate_final = length(vi);   
        figure(2);clf(2); plot(kxz); figure(4); plot(xi); title('xi');     
        fprintf('final: norm(z)=%f num_violations=%d\n', norm(z), length(vi));  
    end    
       
    function obj = objective(M)
        A = M.A; xi = M.S;
        P2 = (K*A)*A'; 
        obj = 0.5*trK + 0.5*(trace((P2 - 2*eye(n))*(P2*K))) + C*sum(xi.^2);
        [vi, vj, vxi] = get_violations(A, xi);        
        if ~isempty(vi)                        
            P3 = (A'*K)*A;                   
            K12 = fast_KK(vi) - fast_KK(vj); 
            %K1 = K(:,vi); K2 = K(:,vj); K12 = K1*K1' - K2*K2';
            obj = obj + 0.5*lambda*(trace((K12*A)*(P3*A'))+ thresh*length(vi) - sum(vxi.^2));            
        end        
    end

    function KK = fast_KK(vv)
        uv = unique(vv); 
        if length(uv)>1, hv = hist(vv, uv)';  else hv = length(vv); end                
        PP = K(uv,:); 
        KK = PP'*(diag(hv)*PP);
    end

    function gM = grad(M)
        A = M.A; xi = M.S;
        P1 = K*(K*A); P2 = (K*A)*A'; P3 = A'*(K*A);
        gA = P1*(P3-2*eye(num_subspaces)) + P2*P1;
        gXi = 2*C*xi;
        [vi,vj, vxi] = get_violations(A, xi);
        if ~isempty(vi)
            %K1 = K(:,vi)*K(:,vi)'; K2 = K(:,vj)*K(:,vj)'; K12 = K1-K2;            
            K12 = fast_KK(vi) - fast_KK(vj);
            gA = gA + lambda*(K12*A*P3 + P2*K12*A);            
            gXi = gXi - 2*lambda*vxi;            
        end        
        gM.A = gA; gM.S = gXi;        
    end

    function [vi, vj, vxi] = get_violations(A, xi) 
       kxz = get_kxz(A*A', K);
       vv = kxz*ones(1,n) - ones(n,1)*kxz' + thresh; %xi-xj+1 i<j
       vv = vv(idx)-xi'.^2;
       ii = idx(vv>0);       
       if any(ii)
           [vi,vj] = ind2sub([n,n], ii);
       else
           [vi,vj] = deal([]);
       end
       vxi = xi; vxi(vv<0) = 0;
    end

   % some easy initializations to be used to compute bandwidth if not
   % supplied
   function xx = initialize_z()
       %xx = X(end,:);
        xx = mean(X,1);        
        %xx = krp_linear_manopt(X, 1/size(X,1), 100, 100, []);          
        %[u,s,v]=svd(cov(X)); xx=u(:,1)';
    end
end
