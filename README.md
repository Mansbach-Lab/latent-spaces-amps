
<H1>Latent Spaces for AMP design</H1>


<h3>A project investigating de novo antimicrobial peptide generation using VAEs and exploring latent spaces</h3>

<h4> Structure of the Git Repo:</h4>

<ul>
  <li> Folders:
    <ul>
      <li> checkpointz: The model checkpoints.
      <li> data: The datasets and data processing notebooks.
      <li> model_analyses: Analyses performed once the training is complete using the checkpoints.
      <li> pickled_pcas: Pickled principal components analysis of all the models. (see corresponding paper)
      <li> scripts: Scripts necessary to train and analyse the models.
      <li> transvae: The models. (including the steadiness and cohesiveness package)
      <li> trials: The output "log" folder when training models.
    </ul>
  <li> Notebooks & Files:
    <ul>
      <li> analysis_of_model_results: Trained model analysis performed over many notebook cells one step at a time.
      <li> notebook_model_training: For training the model locally in a notebook.
      <li> output_graphing_notebook: Use the analysis results to create plots for the paper.
      <li> script_for_combined_model_analysis: Created to submit analysis as a job to a compute cluster on Compute Canada nodes.
      <li> structure-assessment: Notebook to download and view PDB files.
      <li> train_only_requirements: Requirements for training.
      <li> visualizing_attention: Notebook for visualising the attention heads.
    </ul>
</ul>
