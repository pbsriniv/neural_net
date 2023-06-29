clear all;
close all;

rand("state","reset");

##v= rand("state");
##v= ones(size(v))*10;
##rand("state", v);

mnist_neural_func;

dat=load_mnist_images('train-images.idx3-ubyte');
lbl=load_mnist_label('train-labels.idx1-ubyte');

%renormalize the data
%dat = (dat - mean(dat(:)));
%dat = dat / std(dat(:)); % Sets the std-dev about 1.0
%dat = dat; % Scale so we explore the dynamic range of the activation function better
dat = dat - (max(dat(:))+min(dat(:)))/2;
dat = dat / std(dat(:)); % Sets the std-dev about 1.0

%% Test if scanning looks reasonable
for cnt=1:3
 figure; imshow(dat(:,:,cnt)');
 title_str = sprintf("Image is %d",lbl(cnt));
 title(title_str);
end

trn_len=length(lbl);
%% Reformat data for Input
x_dat = reshape(dat,[],trn_len);
z_dat = zeros(10,trn_len);
for cnt=1:trn_len
  z_dat(lbl(cnt)+1,cnt)=1;
endfor


### Use hidden layers suggestions from http://yann.lecun.com/exdb/mnist/
num_layers = 2;
dimn_in = size(x_dat,1);
dimn_out = [500,150,10];  % Should have error rate of 2.95%
dimn_out = [dimn_in*2,10];
%dimn_out = [10];

%% Now train the network
% Init the parameters
layer_dimn_in = dimn_in;
step_a = 0.2;
step_b = 0.1;
for layer = 1:num_layers
  layer_dimn_out = dimn_out(layer);
  a{layer}=2*rand(layer_dimn_out,layer_dimn_in)-1;
  b{layer}=2*rand(layer_dimn_out,1)-1;
  layer_dimn_in = layer_dimn_out;
end;

% train
[a1, b1, err] =  back_prop_vec(x_dat,z_dat,a,b,num_layers,step_a, step_b, 40);

%[a1, b1, err] =  back_prop_vec_t2(x_dat,z_dat,a1,b1,num_layers,step_a, step_b, 10000);
% Verfify convergence
figure;
plot(err,'r');
title('plot of training error vs iteration');

%%Lets verify the Neural Net
dat_orig=load_mnist_images('t10k-images.idx3-ubyte');
lbl=load_mnist_label('t10k-labels.idx1-ubyte');

%renormalize the data
dat = reshape(dat_orig,[],length(lbl));
dat = dat - (max(dat(:))+min(dat(:)))/2;
dat = dat / std(dat(:)); % Sets the std-dev about 1.0

%Run the network. with the trained coefficentrs
[win, ~, ~, ~] = neural_net(dat,a1,b1,num_layers);

%lookup the answer
lk_table = 0:9;
answer =lk_table(win);
answer = reshape(answer,[],1);

%Get the Error
error = lbl - answer;
error(abs(error)>0)=1;
p_success = (length(error)-sum(error))/length(error)

% plot Error persymbol
error_digit=zeros(length(lk_table),1);
tot_digit=zeros(length(lk_table),1);
for cnt=1:length(lbl)
  error_digit(lbl(cnt)+1) += error(cnt);
  tot_digit(lbl(cnt)+1) +=1;
endfor

figure;
bar(lk_table,error_digit./tot_digit);
title("Probability of Error of various digits");

figure;
for cnt = 1:10
 imshow(dat_orig(:,:,cnt)');
 title_str = sprintf("Image is %d, While our NN guessed %d",lbl(cnt),answer(cnt));
 title(title_str);
 fprintf("Paused.. hit enter\n");
 pause
endfor

