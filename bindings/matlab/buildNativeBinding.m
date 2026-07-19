function outputPath = buildNativeBinding()
% Build the DAG-ML task-binding MEX bridge for MATLAB.

bindingRoot = fileparts(mfilename('fullpath'));
sourcePath = fullfile(bindingRoot, 'native', 'task_training_loss_binding.c');
outputDirectory = fullfile(bindingRoot, '+dagml');
mex('-outdir', outputDirectory, '-output', ...
    'taskTrainingLossBindingNative', sourcePath);
outputPath = fullfile(outputDirectory, ...
    ['taskTrainingLossBindingNative.' mexext]);
end
