# python -u allegro_screwdriver.py allegro_screwdriver_diff_init > ./logs/allegro_screwdriver_diff_init.log

# python -u allegro_screwdriver.py allegro_screwdriver_proj_diff_init > ./logs/allegro_screwdriver_proj_diff_init.log
# python -u allegro_screwdriver.py allegro_screwdriver_latent_diff_init_csvto > ./logs/allegro_screwdriver_latent_diff_init_csvto.log

# python -u allegro_screwdriver.py allegro_screwdriver_rand_init > ./logs/allegro_screwdriver_rand_init.log
# python -u allegro_screwdriver.py allegro_screwdriver_diff_init_csvto > ./logs/allegro_screwdriver_diff_init_csvto_pre_sampled.log
# python viz_allegro_plan_init.py allegro_screwdriver_diff_init_csvto
# python viz_allegro_plan_init.py allegro_screwdriver_proj_diff_init_csvto

# python viz_allegro_plan_init.py allegro_screwdriver_diff_init_csvto_high_noise
# python viz_allegro_plan_init.py allegro_screwdriver_proj_diff_init_csvto_high_noise
python -u viz_allegro_plan_init.py allegro_screwdriver_proj_diff_init_csvto_guidance

# python -u allegro_screwdriver.py allegro_screwdriver_latent_diff_init_csvto > ./logs/allegro_screwdriver_latent_diff_init_csvto.log

python -u viz_allegro_plan_init.py allegro_screwdriver_diff_init_csvto_guided
