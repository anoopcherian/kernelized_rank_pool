#kernelized_rank_pool
Implementation of Kernelized Rank Pooling variants. If you use this code,
please cite our paper:

 <b>Non-Linear Temporal Subspace Representations for Activity Recognition, 
 A. Cherian, S. Sra, S. Gould, and R. Hartley, CVPR 2018. </b>
 
 This code implements:
 1. Basic Kernelized Rank Pooling (BKRP) (use algo_type=RBF_WithSlack)
 2. Improved Basic Kernelized Rank Pooling (IBKRP) (use algo_type='RBF-XRecon-WithSlack')
 3. Kernelized Rank Pooling with Feature Subspaces (KRP-FS) (use algo_type='RBF-FS_WithSlack')
 4. Rank Pooling (Video Darwin) (use algo_type='rankpool')
 5. Linear (Video Darwin, but implemented using a linear kernel) (use algo_type='linear')
 6. Misc: There are also several other variants of pooling in krp folder (undocumented and unpublished, see code).


 For bugs and comments, email Anoop Cherian at anoop.cherian@gmail.com
 Last modified: 14th June 2018


 Copyright (c) 2018, Anoop Cherian
 All rights reserved.
 Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
 INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY 
 OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT 
 OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 
 The following code shows a demo implementation and usage of KRP schemes
 on the JHMDB dataset split 1. JHMDB uses VGG-16 fc6 features (available
 in the bundle). Data for HMDB dataset ResNet-152 two stream features are
 available separately.

 The code has the following dependencies which you have to download separately and unzip to appropriate folders. 
 1. LibLinear version 2: 
 2. ManOpt Package
 3. LibSVM version 3.14
 4. JHMDB dataset.
 All the above are available to download from http://users.cecs.anu.edu.au/~cherian/code/krp_fs.zip.
 
 A link to <b>HMDB dataset ResNet-152 features</b> for RGB and Flow are available at http://users.cecs.anu.edu.au/~cherian/
 
 How to run the code? Just run demo_krp.m (after fixing the dependencies). The code runs KRP on JHMDB dataset split 1. Tested on Matlab 2018.
