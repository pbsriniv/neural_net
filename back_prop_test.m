clear all;
close all;

rand("state","reset");

##v= rand("state");
##v= ones(size(v))*10;
##rand("state", v);

mnist_neural_func;

% Generate data to test my network
  num_layers = 2;
  dimn_in = 4;
  %dimn_out = round(rand(num_layers,1)*10 + 1);
  dimn_out = [3,2];

[z_dat, x_dat, coeff, bias] = gen_test_data(num_layers, dimn_in, dimn_out);


%% Now train the network

% Init the parameters
layer_dimn_in = dimn_in;
step = 0.05;
for layer = 1:num_layers
  layer_dimn_out = dimn_out(layer);
  a{layer}=randn(layer_dimn_out,layer_dimn_in);

  #a{2}=coeff{2} + rand(size(coeff{2}))*0.0001;
  #a{1}=coeff{1} + rand(size(coeff{1}))*0.0001;
  b{layer}=zeros(layer_dimn_out,1);
  layer_dimn_in = layer_dimn_out;
end;



##[a, b, err] =  back_prop(x_dat,z_dat,a,b,num_layers,step);
[a1, b1, err] =  back_prop_vec(x_dat(:,1:end),z_dat(:,1:end),a,b,num_layers,step,100);
#fprintf("------------------");
#[a2, b3, err] =  back_prop_vec_test(x_dat(:,1:end),z_dat(:,1:end),a,b,num_layers,step,10);


figure;
plot(err,'r');

figure;
plot((coeff{1}(:) - a1{1}(:))./coeff{1}(:),'r'); hold on;
plot((coeff{2}(:) - a1{2}(:))./coeff{2}(:),'b');


##coeff{2}(:)
##a{2}(:)

##plot(err(1,1,:)- coeff(1,1),'r'); hold on;
##plot(err(1,2,:)- coeff(1,2),'b');
##plot(err(1,3,:)- coeff(1,3),'g');




