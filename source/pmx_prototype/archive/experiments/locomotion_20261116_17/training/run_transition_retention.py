"""Bounded transition coverage and frozen-policy retention continuation."""
import os
import continue_sampling_pivot as workflow

workflow.OUT=workflow.ROOT/'output/anon_locomotion/transition_retention_20261116'
workflow.CONFIG=workflow.OUT/'prepared_config.json'
workflow.STAGE_ITERATIONS=[100,100,100]
workflow.CUMULATIVE_BASE=48054.58603930008
workflow.EXACT_PREFIX_CONFIG=workflow.ROOT/'output/anon_locomotion/reverse_turn_diagnosis_20261116/prefix_config.json'

if __name__=='__main__':
    os.environ['OMNI_KIT_ACCEPT_EULA']='YES'
    try:workflow.main()
    except Exception as error:
        workflow.write(workflow.OUT/'status.json',dict(stage='error',error=str(error)));raise
