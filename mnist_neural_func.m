1;

function lbl = load_mnist_label(fname)
  fpr=fopen(fname,"r");
  [magic_num, cnt] = fread(fpr,1,"uint32",0,"ieee-be");
  [num_items, cnt] = fread(fpr,1,"uint32",0,"ieee-be");
  [lbl, cnt] = fread(fpr,inf,"uint8",0,"ieee-be");
endfunction

function val = load_mnist_images(fname)
  fpr=fopen(fname,"r");
  [magic_num, cnt] = fread(fpr,1,"uint32",0,"ieee-be");
  [num_items, cnt] = fread(fpr,1,"uint32",0,"ieee-be");
  [row_col, cnt] = fread(fpr,2,"uint32",0,"ieee-be");

  exp_size= [row_col; num_items];

  [val, cnt] = fread(fpr,inf,"uint8",0,"ieee-be");
  val= reshape(val,exp_size);
endfunction

function val = activ_func(x)
  %val = tanh(x);
  val = (1 ./ (1+exp(-x/5)));
endfunction

function val = diff_activ_func(x)
  %val = 1.0 - tanh(x).^2;
  val = (activ_func(x)) .* (1 - activ_func(x));
endfunction

function [win, ox_z, ox_y, ox_x] = neural_net(x,a,b,num_layers)
  inp_layer = x;

  ox_x={};
  ox_y={};
  ox_z={};

  for layer = 1:num_layers
    y = a{layer} * inp_layer + b{layer};
    z = activ_func(y);

    %Store for function outputs
    ox_x{layer}=inp_layer;
    ox_y{layer} =y;
    ox_z{layer} =z;

    %Set the input for the next layer
    inp_layer = z;
  endfor
 [~, win] = max(ox_z{num_layers},[],1);
endfunction


function [a, b, error] = back_prop(inp,tru,a,b,num_layers,step)
  num_train = length(inp);
  error = zeros(num_train,1);

  for vec_cnt = 1:num_train
  % First compute the outputs for each layers
  [~, ox_z, ox_y, ox_x] = neural_net(inp(:,vec_cnt),a,b,num_layers);

  common_fac = -2*(tru(:,vec_cnt) - ox_z{num_layers})';
  ## back propagate
  for layer = num_layers: -1 : 1
      common_fac = common_fac  * diag(diff_activ_func_func(ox_y{layer}));

      % Take advantage of the fact that most of the matrix rows are zeros, except the
      % row of interest.. so do a pointwise multiplication to generate all the ouputs
      % at once instead of looping multiple times
     grad_x =  (common_fac') .* (repmat(ox_x{layer}',length(ox_y{layer}),1));
     grad_b = (common_fac') .* ones(length(o_y{layer}),1);

##      grad_x = (common_fac') * ox_x{layer}';
##      grad_b = (common_fac') * 1;

    %% iterate for the next layer
    common_fac  = common_fac * a{layer};
    a{layer} = a{layer} - step*grad_x; % correct the coefficient
    b{layer} = b{layer} - step*grad_b; % correct the coefficient
  endfor

  %tru(:,vec_cnt) - ox_z{num_layers};
  error(vec_cnt) = (tru(:,vec_cnt) - ox_z{num_layers})' * (tru(:,vec_cnt) - ox_z{num_layers});
  %error(:,:,vec_cnt) = a{num_layers};
  endfor
endfunction


function [a, b, error] = back_prop_vec(inp,tru,a,b,num_layers,step_coef, step_bias,trn_size)
  num_pass = 20;
  error = [];

  for pass = 1:num_pass
  trn_size_pass  = trn_size*pass;
  num_train = ceil(length(inp)/trn_size_pass);
  error1 = zeros(num_train,1);

  for grad_cnt = 1 : num_train
    inp_train   = inp(:,(grad_cnt-1)*trn_size_pass+1:min(grad_cnt*trn_size_pass,length(inp)));
    truth_train = tru(:,(grad_cnt-1)*trn_size_pass+1:min(grad_cnt*trn_size_pass,length(inp)));

    vec_size = size(truth_train,2);
    % Handle a vector of Data

    % First compute the outputs for each layers
    [~, ox_z, ox_y, ox_x] = neural_net(inp_train,a,b,num_layers);
    common_fac = -2*(truth_train - ox_z{num_layers});
  ## back propagate
  for layer = num_layers: -1 : 1
      common_fac = common_fac  .* diff_activ_func(ox_y{layer}); % multiply each column
      grad_x =  (common_fac * ox_x{layer}')/vec_size; % sum the along the vector length and average with the vector multiply
      grad_b = (common_fac * ones(vec_size,1))/vec_size; % sum along the vector

    %% iterate for the next layer, apply the coefficient on each vector
    common_fac  = (common_fac' * a{layer})';
    %grad_x
    %grad_b
    a{layer} = a{layer} - step_coef*grad_x; % correct the coefficient
    b{layer} = b{layer} - step_bias*grad_b; % correct the coefficient
  endfor

  %tru(:,vec_cnt) - ox_z{num_layers};
  error1(grad_cnt) =sum(sum((truth_train - ox_z{num_layers}).^2,2),1)/vec_size;

  %error(:,:,vec_cnt) = a{num_layers};
endfor
  error = [error; error1]; % append
endfor

  % First compute the outputs for each layers
  [~, ox_z, ox_y, ox_x] = neural_net(inp(:,1:5),a,b,num_layers);
  ox_z{num_layers}
  tru(:,1:5)

endfunction

##----------------------------------------------------------
## Test code
##----------------------------------------------------------

function [a, b, error] = back_prop_vec_test(inp,tru,a,b,num_layers,step,trn_size)
  num_train = length(inp);
  error = zeros(num_train/trn_size,1);

  for vec_cnt_out = 1:1 %num_train/trn_size
   grad_x = {zeros(2), zeros(2)};
   grad_b = {zeros(2,1), zeros(2,1)};
   error_acc = 0;
   for vec_cnt_in = 1:trn_size
     vec_cnt = (vec_cnt_out-1)*trn_size+vec_cnt_in;

    % First compute the outputs for each layers
    [~, ox_z, ox_y, ox_x] = neural_net(inp(:,vec_cnt),a,b,num_layers);
    common_fac = -2*(tru(:,vec_cnt) - ox_z{num_layers})';
    error_acc = error_acc+(tru(:,vec_cnt) - ox_z{num_layers})' * (tru(:,vec_cnt) - ox_z{num_layers});

    ## back propagate
    for layer = num_layers: -1 : 1
      common_fac = common_fac  * diag(diff_sigma_func(ox_y{layer}));

      % Take advantage of the fact that most of the matrix rows are zeros, except the
      % row of interest.. so do a pointwise multiplication to generate all the ouputs
      % at once instead of looping multiple times
      grad_x{layer} +=  (common_fac') .* (repmat(ox_x{layer}',length(ox_y{layer}),1));
      grad_b{layer} +=  (common_fac') .* ones(length(ox_y{layer}),1);

      %% iterate for the next layer
      common_fac  = common_fac * a{layer};
    endfor
  endfor

  for layer = num_layers: -1 : 1
    grad_x_tmp = grad_x{layer}/trn_size
    grad_b_tmp = grad_b{layer}/trn_size

    a{layer} = a{layer} - step*grad_x_tmp; % correct the coefficient
    b{layer} = b{layer} - step*grad_b_tmp; % correct the coefficient
  endfor

  error(vec_cnt_out) =error_acc/trn_size;
  endfor
endfunction


function [z_dat, x_dat, a, b] = gen_test_data(num_layers, dimn_in, dimn_out)

% Initialize
a = {};
b = {};

dimn_prev = dimn_in;
for layer=1:num_layers
  a{layer} = 2*(rand(dimn_out(layer), dimn_prev)-0.5);
  b{layer} = zeros(dimn_out(layer),1);
  dimn_prev = dimn_out(layer);
end;

  vect_len = 100000;
  x_dat = randn(dimn_in, vect_len);

  %Actual
  [~, ox_z, ~, ~] = neural_net(x_dat,a,b,num_layers);
  z_dat = ox_z{num_layers};

endfunction

