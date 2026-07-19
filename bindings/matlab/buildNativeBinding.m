function outputPath = buildNativeBinding()
% Build the DAG-ML native MEX bridges for MATLAB.

bindingRoot = fileparts(mfilename('fullpath'));
outputDirectory = fullfile(bindingRoot, '+dagml');

taskSourcePath = fullfile(bindingRoot, 'native', 'task_training_loss_binding.c');
mex('-outdir', outputDirectory, '-output', ...
    'taskTrainingLossBindingNative', taskSourcePath);
phaseSourcePath = fullfile(bindingRoot, 'native', 'execution_plan_phase.c');
mex('-outdir', outputDirectory, '-output', ...
    'executeExecutionPlanPhaseNative', phaseSourcePath);

outputPath = { ...
    fullfile(outputDirectory, ['taskTrainingLossBindingNative.' mexext]), ...
    fullfile(outputDirectory, ['executeExecutionPlanPhaseNative.' mexext]) ...
    };
end
