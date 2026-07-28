"""Per-environment defaults for the online entry-point scripts.

The `*_final` wandb runs each needed ~25 CLI flags because every per-task value
(gin file, cond_top_frac, accumulation_steps, alpha_rtb, ...) was passed by hand.
That table now lives here, so a launch command only has to say *which task* and
*which algorithm*:

    python synther/online/online_cond_ddpm_ori.py --env hopper --seed 0 --wandb

Anything passed explicitly on the command line still wins -- the entry points give
these arguments a default of ``None`` and `apply_env_defaults` only fills the holes.

Values reproduce the runs recorded in exp/01_experiments.md, with one deliberate
change: HalfCheetah-v2 now uses sac_cond_synther_openai.gin (it was run with
dmc.gin, which left reward normalization off -- see exp/02_code_notes.md I-5).
"""

DMC_GIN = 'config/online/sac_cond_synther_dmc.gin'
OPENAI_GIN = 'config/online/sac_cond_synther_openai.gin'

# Short names accepted by --env, matching the wandb project names.
ENV_ALIASES = {
    'quad': 'quadruped-walk-v0',
    'cheetah': 'cheetah-run-v0',
    'reacher': 'reacher-hard-v0',
    'fingereasy': 'finger-turn_easy-v0',
    'finger_easy': 'finger-turn_easy-v0',
    'fingerhard': 'finger-turn_hard-v0',
    'finger_hard': 'finger-turn_hard-v0',
    'hopper': 'Hopper-v2',
    'walker': 'Walker2d-v2',
    'half': 'HalfCheetah-v2',
    'halfcheetah': 'HalfCheetah-v2',
}

# epochs                : total SAC/REDQ epochs (1000 env steps each)
# cond_top_frac         : top-novelty fraction used as the CFG conditioning signal
# accumulation_steps    : gradient accumulation during RTB fine-tuning
# num_posterior_epochs  : RTB fine-tuning epochs per diffusion retrain
# ft_clip_grad          : RTB grad-norm clip; 0.0 disables clipping
# alpha_rtb             : RTB temperature, per novelty measure
ENV_CONFIG = {
    'quadruped-walk-v0': dict(
        gin=DMC_GIN, epochs=100, cond_top_frac=0.1,
        accumulation_steps=8, num_posterior_epochs=150, ft_clip_grad=0.0,
        alpha_rtb={'curiosity': 1.0, 'rnd': 10.0},
    ),
    'cheetah-run-v0': dict(
        gin=DMC_GIN, epochs=100, cond_top_frac=0.25,
        accumulation_steps=4, num_posterior_epochs=100, ft_clip_grad=1.0,
        alpha_rtb={'curiosity': 10.0, 'rnd': 4.0},
    ),
    'reacher-hard-v0': dict(
        gin=DMC_GIN, epochs=100, cond_top_frac=0.25,
        accumulation_steps=4, num_posterior_epochs=100, ft_clip_grad=1.0,
        alpha_rtb={'curiosity': 10.0, 'rnd': 4.0},
    ),
    'finger-turn_easy-v0': dict(
        gin=DMC_GIN, epochs=300, cond_top_frac=0.25,
        accumulation_steps=4, num_posterior_epochs=100, ft_clip_grad=0.0,
        alpha_rtb={'curiosity': 2.0, 'rnd': 4.0},
    ),
    'finger-turn_hard-v0': dict(
        gin=DMC_GIN, epochs=300, cond_top_frac=0.25,
        accumulation_steps=4, num_posterior_epochs=100, ft_clip_grad=0.0,
        alpha_rtb={'curiosity': 10.0, 'rnd': 10.0},
    ),
    'Hopper-v2': dict(
        gin=OPENAI_GIN, epochs=100, cond_top_frac=0.25,
        accumulation_steps=4, num_posterior_epochs=100, ft_clip_grad=1.0,
        alpha_rtb={'curiosity': 2.0, 'rnd': 1.0},
    ),
    'Walker2d-v2': dict(
        gin=OPENAI_GIN, epochs=100, cond_top_frac=0.25,
        accumulation_steps=4, num_posterior_epochs=100, ft_clip_grad=1.0,
        alpha_rtb={'curiosity': 0.5, 'rnd': 2.0},
    ),
    'HalfCheetah-v2': dict(
        gin=OPENAI_GIN, epochs=100, cond_top_frac=0.25,
        accumulation_steps=4, num_posterior_epochs=100, ft_clip_grad=1.0,
        alpha_rtb={'curiosity': 2.0, 'rnd': 2.0},
    ),
}

# Not part of the `*_final` tables, but the previous hardcoded rule gave these 300
# epochs alongside finger-turn_*, so keep that. RTB values are untuned guesses.
for _humanoid in ('humanoid-run-v0', 'humanoid-walk-v0'):
    ENV_CONFIG[_humanoid] = dict(
        gin=DMC_GIN, epochs=300, cond_top_frac=0.25,
        accumulation_steps=4, num_posterior_epochs=100, ft_clip_grad=1.0,
        alpha_rtb={'curiosity': 1.0, 'rnd': 1.0},
    )

# Fallback for an env that is not in the table at all: DMC-style setup.
FALLBACK = dict(
    gin=DMC_GIN, epochs=100, cond_top_frac=0.25,
    accumulation_steps=4, num_posterior_epochs=100, ft_clip_grad=1.0,
    alpha_rtb={'curiosity': 1.0, 'rnd': 1.0},
)


def resolve_env(name):
    """Expand a short alias (`half`) to the gym id (`HalfCheetah-v2`)."""
    return ENV_ALIASES.get(name, name)


def apply_env_defaults(args, rtb=False):
    """Fill every `None` argument from the per-env table. Explicit flags win.

    `rtb=True` also resolves the RTB fine-tuning arguments, which only the
    Ours entry point has.  Returns `args` (mutated in place) so it can be
    used inline.
    """
    args.env = resolve_env(args.env)
    cfg = ENV_CONFIG.get(args.env)
    if cfg is None:
        print(f'[env_defaults] "{args.env}" is not in ENV_CONFIG; using FALLBACK.')
        cfg = FALLBACK

    if not args.gin_config_files:
        args.gin_config_files = [cfg['gin']]
    for key in ('epochs', 'cond_top_frac'):
        if getattr(args, key, None) is None:
            setattr(args, key, cfg[key])

    if rtb:
        for key in ('accumulation_steps', 'num_posterior_epochs', 'ft_clip_grad'):
            if getattr(args, key, None) is None:
                setattr(args, key, cfg[key])
        if getattr(args, 'alpha_rtb', None) is None:
            table = cfg['alpha_rtb']
            if args.novelty_measure not in table:
                print(f'[env_defaults] no tuned alpha_rtb for '
                      f'novelty_measure={args.novelty_measure} on {args.env}; '
                      f'falling back to the curiosity value.')
            args.alpha_rtb = table.get(args.novelty_measure, table['curiosity'])

    resolved = ['env', 'gin_config_files', 'epochs', 'cond_top_frac']
    if rtb:
        resolved += ['accumulation_steps', 'num_posterior_epochs',
                     'ft_clip_grad', 'alpha_rtb']
    print('[env_defaults] ' + '  '.join(f'{k}={getattr(args, k)}' for k in resolved))
    return args
