% Implementation of Kernelized Rank Pooling variants. If you use this code,
% please cite our paper:
%
% Non-Linear Temporal Subspace Representations for Activity Recognition, 
% A. Cherian, S. Sra, S. Gould, and R. Hartley, CVPR 2018.
% 
% This code implements:
% 1. Basic Kernelized Rank Pooling (BKRP) (use algo_type=RBF_WithSlack)
% 2. Improved Basic Kernelized Rank Pooling (IBKRP) (use algo_type='RBF-XRecon-WithSlack')
% 3. Kernelized Rank Pooling with Feature Subspaces (KRP-FS) (use algo_type='RBF-FS_WithSlack')
% 4. Rank Pooling (Video Darwin) (use algo_type='rankpool')
% 5. Linear (Video Darwin, but implemented using a linear kernel) (use algo_type='linear')
% 6. Misc: There are also several other variants of pooling in krp folder (undocumented and unpublished, see code).
%

% For bugs and comments, email Anoop Cherian at anoop.cherian@gmail.com
% Last modified: 14th June 2018
%

% Copyright (c) 2018, Anoop Cherian
% All rights reserved.
% Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
% 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
% 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
% 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
% PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
% INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
% SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY 
% OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT 
% OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
% 
% The following code shows a demo implementation and usage of KRP schemes
% on the JHMDB dataset split 1. JHMDB uses VGG-16 fc6 features (available
% in the bundle). Data for HMDB dataset ResNet-152 two stream features are
% available separately.

function demo_krp(algo_type, num_subspaces, rank_thresh)
addpath ./tools/dists;
addpath ./tools/svm/liblinear2/matlab;
addpath ./tools/svm/libsvm-3.14/matlab;
addpath(genpath('./tools/manopt/'));
addpath(genpath('./krp/'));

randn('state',1234); rand('twister', 1234);

accuracy = [];
frame_skip_interval = 10000; % use 2 for HMDB if the number of frames is too much per sequence and you want to skip every other.
num_iter = 50; % number of iterations for the Riemannian descent optimization.
split_num = 1; % the split number of use for JHMDB.
feat_dim = 4096; % JHMDB we use fc6 of VGG network. We assume each video sequence is encoded as a dxn matrix with d as the feature dim and n frames.

if nargin == 0
	%split_num = 1; algo_type ='RBF-FS-WithSlack'; num_subspaces = 5; rank_thresh=0.001;
    fprintf('----------------------------------------------------------------\n');
    fprintf('USAGE: demo_krp(algo_type, num_subspaces, rank_thresh)\n')
    fprintf('algo_type could be: \n \t RBF-WithSlack (for BKRP) \n \t RBF-XRecon-WithSlack (for IBKRP) \n \t RBF-FS-WithSlack (for KRP_FS)\n \t RankPool (for VideoDarwin)\n');
    fprintf('example usage: demo_krp(''RBF-FS-WithSlack'', 5, 0.001)\n');
    fprintf('----------------------------------------------------------------\n');
    fprint('Using default settings and JHMDB dataset split#1\n')
    algo_type='RBF-FS-WithSlack'; num_subspaces=5; rank_thresh=0.001;    
end

for splits=split_num
    fprintf('Working on JHMDB dataset split %d: num_subspaces=%d algo_type=%s rank_thresh=%f num_frames_per_seq=%d\n', ...
            splits, num_subspaces, algo_type, rank_thresh, frame_skip_interval);
        
    fprintf('loading JHMDB split_1 data...\n');
    load(['./data/JHMDB_split' num2str(splits) '.mat']);
    
    % sequence ids to use.
    train_seq_ids = JHMDB.train_idx; 
    test_seq_ids = JHMDB.test_idx; 
        
    % preprocessing
    imflow = preprocess_data(JHMDB.flow_data, frame_skip_interval, feat_dim);
    imdata = preprocess_data(JHMDB.rgb_data, frame_skip_interval, feat_dim);  

     % compute grassmannian descriptors: Code produces descriptors for
     % forward and reversed sequences.     
    fprintf('compute KRP descriptors for flow...\n'); 
    [fwd, bwd] = get_KRP_descriptors(imflow, algo_type, num_subspaces, rank_thresh, num_iter);    
    imflow_desc{1} = fwd;
    if ~isempty(bwd)
        imflow_desc{2} = bwd;
    end    

     fprintf('compute KRP descriptors for rgb\n'); 
     [fwd, bwd] = get_KRP_descriptors(imdata, algo_type, num_subspaces, rank_thresh, num_iter);
     imdata_desc{1} = fwd;
     if ~isempty(bwd)
        imdata_desc{2} = bwd;
     end    
    
    all_labels = JHMDB.labels; 
  
    [A_train_imdata, A_train_imflow, A_test_imdata, A_test_imflow] = deal({});
    [X_train_imdata, X_train_imflow, X_test_imdata, X_test_imflow] = deal({});
    
    A_train_imdata{1} = imdata_desc{1}(train_seq_ids); % fwd    
    A_test_imdata{1}  = imdata_desc{1}(test_seq_ids);    
    A_train_imdata{2} = imdata_desc{2}(train_seq_ids); % bwd
    A_test_imdata{2}  = imdata_desc{2}(test_seq_ids);
    
    X_train_imdata{1} = imdata(train_seq_ids);     
    X_test_imdata{1} = imdata(test_seq_ids);    
    X_train_imdata{2} = create_bwd(imdata(train_seq_ids)); 
    X_test_imdata{2} = create_bwd(imdata(test_seq_ids));
    
    A_train_imflow{1} = imflow_desc{1}(train_seq_ids);
    A_test_imflow{1}  = imflow_desc{1}(test_seq_ids);
    A_train_imflow{2} = imflow_desc{2}(train_seq_ids);
    A_test_imflow{2}  = imflow_desc{2}(test_seq_ids);
    
    X_train_imflow{1} = imflow(train_seq_ids);
    X_test_imflow{1} = imflow(test_seq_ids);
    X_train_imflow{2} = create_bwd(imflow(train_seq_ids));
    X_test_imflow{2} = create_bwd(imflow(test_seq_ids));

    encoded_train_labels = all_labels(train_seq_ids); 
    encoded_test_labels = all_labels(test_seq_ids);   
   
    % classify and evaluate.
    if strcmp(algo_type, 'RBF-FS-WithSlack') % KRP FS uses exponential projection metric kernel.
         [model,a,b,c] = train_and_test_svm_FS(A_train_imdata, A_train_imflow, A_test_imdata, A_test_imflow, ...
                                            X_train_imdata, X_train_imflow, X_test_imdata, X_test_imflow, ...
                         encoded_train_labels, encoded_test_labels, num_subspaces); 
                     
    else % other kernel pooling methods use a linear SVM kernel.
         X_flow = [cell2mat(imflow_desc{1}'), cell2mat(imflow_desc{2}')]; 
         X_rgb  = [cell2mat(imdata_desc{1}'), cell2mat(imdata_desc{2}')];
         X = [X_flow, X_rgb/8]; % reduce the weight of rgb, helps!
         [model,a,b,c] = train_and_test_svm_KRP(X(train_seq_ids,:), X(test_seq_ids,:), encoded_train_labels, encoded_test_labels);
    end
                                      
    accuracy(splits) = b(1); 
    fprintf('accuracy-split%d = %f\n', splits, b(1));
end
str = sprintf('average classification accuracy for %d splits = %f\n', splits, mean(accuracy));
fprintf('%s', str);
end

     
function [fwd, bwd] = get_KRP_descriptors(xx, manifold_type, num_subspaces, thresh, num_iter)
    fwd = []; bwd = [];
    if strcmp(manifold_type, 'RankPool')
        for t=1:length(xx)
            fwd{t} = vec(VideoDarwinRepresentation(xx{t}',1))';
            bwd{t} = fwd{t};
        end
     
	elseif strcmp(manifold_type, 'RBF-FS-WithSlack')
		sigma = 1; lambda = 1; num_iter = 100; 
		[fwd, bwd] = KRP_FS_wrapper(xx, num_subspaces , manifold_type, thresh, num_iter, sigma, lambda); 
        
    else %if strcmp(manifold_type, 'RBF-XRecon-WithSlack') || strcmp(manifold_type, 'RBF-WithSlack')  % IBKRP/BKRP
         % sigma is the variance of the RBF kernel (if null estimated automatically, 
         % lambda is the penalty for the enforcing the temporal ordering.
         % num_iter is the number of iterations for the Riemannian solver.
         % thresh is the \eta in the paper -- the margin for the temporal ordering constraint.
		sigma = []; lambda = 100; num_iter = 100; 
		[fwd, bwd] = KRP_FS_wrapper(xx,  num_subspaces , manifold_type, thresh, num_iter, sigma, lambda); 
	end
end
          
function [fwd, bwd] = KRP_FS_wrapper(seq, num_subspaces, manifold_type,  thresh, num_iter, sigma, lambda)
ratio=zeros(length(seq),1); 
n = size(seq{1},1);
fprintf('We are using sigma=%f thresh=%f num_iter=%d lambda=%f for pooling\n', sigma, thresh, num_iter, lambda);
parfor tt=1:length(seq)    
    xx = normc(double(seq{tt}));       
	try
	  switch manifold_type
        case {'RBF-XRecon-WithSlack', 'linear', 'RBF-Mult-WithSlack', 'RBF-WithSlack'}
            pool_C = VideoDarwinRepresentation_KRP(xx', manifold_type, num_subspaces, thresh); 
            fwd{tt} = pool_C(1:n); bwd{tt} = pool_C(n+1:2*n);

        case 'RBF-FS-WithSlack'
            if size(xx,2)<num_subspaces, nn = size(xx,2); else nn = num_subspaces; end
            pool_C = VideoDarwinRepresentation_FS(xx', manifold_type, nn, thresh, num_iter, sigma); 
            fwd{tt} = pool_C{1}; bwd{tt} = pool_C{2};

        case 'RBF-WithoutSlack'
            [fwd{tt}, ratio(tt)] = krp_rbf_manopt(xx', thresh, lambda, num_iter, sigma);        
   	  end
	catch
      fprintf(['sequence: %d errored out!... continuing...: size(seq) = [%d %d]\n'], tt, size(xx));
	  fwd{tt} = []; bwd{tt} = [];	
	end 

	if mod(tt,50)==0
		fprintf('.');
	end
end
end

function [model, a,b,c] = train_and_test_svm_KRP(train_data, test_data, train_labels, test_labels)        
    n = size(train_data,2);
	train_data = double(train_data); test_data = double(test_data);
 	
    params = [' -q -c 1 -s 2 -n 12 '];         

    model = train(train_labels', sparse((sqrt(abs(train_data)))), params); % using sqrt(abs()) gives better performance apparently!
    [a,b_train,c]= predict(train_labels', sparse((sqrt(abs(train_data)))), model, '-q');
    [a,b,c] = predict(test_labels', sparse((sqrt(abs(test_data)))), model, '-q');

    c(:,model.Label) = c;               
    fprintf('test_accuracy=%f train_accuracy=%f\n', b(1), b_train(1));              
end

function [model,a,b,c] = train_and_test_svm_FS(A_train_imdata, A_train_imflow, A_test_imdata, A_test_imflow, ...
                        train_imdata, train_imflow, test_imdata, test_imflow, ...
                        train_labels, test_labels, num_subspaces)   
    sigma = 1; 
    num_blocks = 2; % use 10 for HMDB; number of parallel blocks in kernel computations (see compute_FS_kernel.m)
    nystrom_ratio=1; % may be more than 1 for approximations. Use 4 for HMDB.
    
    fprintf('computing FS kernel-rgb\n');
    if ~isempty(train_imdata{1})
       fprintf('computing grassmannian kernel for RGB train (fwd)...\n');
       P_tr_imdata_fwd = compute_FS_kernel4(A_train_imdata{1}, train_imdata{1}, ...
            A_train_imdata{1}, train_imdata{1}, sigma, 'train', 'rgb-fwd', 'jhmdb',...
            num_subspaces, nystrom_ratio, num_blocks);
        
       fprintf('computing grassmannian kernel for RGB test (fwd)...\n');
       P_te_imdata_fwd = compute_FS_kernel4(A_test_imdata{1}, test_imdata{1}, ...
           A_train_imdata{1}, train_imdata{1}, sigma, 'test', 'rgb-fwd', 'jhmdb', ...
           num_subspaces, nystrom_ratio, num_blocks);
       
       %P_tr_imdata_bck = compute_FS_kernel3(A_train_imdata{2}, train_imdata{2}, ...
            %A_train_imdata{2}, train_imdata{2}, sigma, 'train', 'rgb-bck', 'jhmdb', ...
            %num_subspaces, nystrom_ratio, num_blocks);
            
       %P_te_imdata_bck = compute_FS_kernel3(A_test_imdata{2}, ...
            %test_imdata{2}, A_train_imdata{2}, train_imdata{2}, sigma, 'test', 'rgb-bck', ...
            %'jhmdb', num_subspaces, nystrom_ratio, num_blocks);
    end

    fprintf('computing FS kernel-flow\n');
    if ~isempty(train_imflow{1})
        fprintf('computing grassmannian kernel for FLOW train (fwd)...\n');
        P_tr_imflow_fwd = compute_FS_kernel4(A_train_imflow{1}, train_imflow{1}, ...
            A_train_imflow{1}, train_imflow{1}, sigma, 'train', 'flow-fwd', 'jhmdb', ...
            num_subspaces, nystrom_ratio, num_blocks);        
        
        fprintf('computing grassmannian kernel for FLOW test (fwd)...\n');
        P_te_imflow_fwd = compute_FS_kernel4(A_test_imflow{1}, test_imflow{1},...
            A_train_imflow{1}, train_imflow{1}, sigma, 'test', 'flow-fwd', 'jhmdb', ...
            num_subspaces, nystrom_ratio, num_blocks);
        
        %P_tr_imflow_bck = compute_FS_kernel3(A_train_imflow{2}, train_imflow{2}, ...
            %A_train_imflow{2}, train_imflow{2}, sigma, 'train', 'flow-bck', 'jhmdb', ...
            %num_subspaces, nystrom_ratio, num_blocks);  

        %P_te_imflow_bck = compute_FS_kernel3(A_test_imflow{2}, test_imflow{2}, ...
            %A_train_imflow{2}, train_imflow{2}, sigma, 'test', 'flow-bck', 'jhmdb', ...
            %num_subspaces, nystrom_ratio, num_blocks);
    end
    fprintf('training SVM \n');
        
    if ~isempty(train_imdata{1})
        P_tr_imdata =  P_tr_imdata_fwd; % + P_tr_imdata_bck; 
        P_te_imdata =  P_te_imdata_fwd; % + P_te_imdata_bck; 
    else
        P_tr_imdata = 0;
        P_te_imdata = 0;
    end

    if ~isempty(train_imflow{1})
        P_tr_imflow =  P_tr_imflow_fwd; % + P_tr_imflow_bck; 
        P_te_imflow =  P_te_imflow_fwd; % + P_te_imflow_bck;
    else
        P_tr_imflow = 0;
        P_te_imflow = 0;
    end

    assert(~isempty(train_imdata{1}) || ~isempty(train_imflow{1}))
    
    params = ' ';        
    params = [params, ' ']; 
    pos_C = 16; g1 = 4;
            
    K_train = [(1:length(train_labels))',  P_tr_imflow + P_tr_imdata/g1]; 
    K_test = [(1:length(test_labels))',    P_te_imflow + P_te_imdata/g1]; 

    model = svmtrain(train_labels', sparse(double(K_train)), ['-q -t 4 -c ' num2str(pos_C) params]);                                
    [~,b_train,~]= svmpredict(train_labels', sparse(double(K_train)), model);
    [a,b,c] = svmpredict(test_labels', sparse(double(K_test)), model);
    fprintf('test_accuracy=%f train_accuracy=%f\n',pos_C, g1, b(1), b_train(1)); 
end

function bwd = create_bwd(X)
    bwd = cellfun(@(x) x(:,end:-1:1), X,'uniformoutput', false);
end

% frame_skip_interval is the num_frames to skip in case it takes too much
% time.
function data = preprocess_data(data, frame_skip_interval, feat_dim)
    xx = data;                                                            
    for t=1:length(xx)   
        nn=size(xx{t},2); 
        ii=1:max(1,round(nn/frame_skip_interval)):nn;         
        xx{t}=normc(xx{t}(1:feat_dim,ii));                                                                        
    end                                                                                           
    data = xx;
end

function X=normc(X)
   X = bsxfun(@rdivide, X, sqrt(sum(X.^2,1))+eps);
end