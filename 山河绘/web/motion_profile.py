"""Color motion limits, independent of the working black/title executor.

Travel limits never raise a source speed. Paper drawing receives a 2.4x scale
after its original cap, plus an optional per-artwork multiplier; commands stay
at or below 100. These are command ratios, not measured speeds.
Web color stages inherit controller acceleration; zero means no AccJ/AccL command.
"""
import copy

DEFAULTS = dict(format='shanhe-web-motion-v1', global_speed_factor=70,
                draw=45, horizontal=80, approach=60, touchdown=30, lift=60,
                dip=20, joint_acceleration=0, linear_acceleration=0)
# Keep accepting older saved records; cap approach/loading at preparation time
# so a still-running server cannot restore the former faster contact speeds.
CONTACT_SPEED_LIMITS = dict(horizontal=40, approach=30, touchdown=15, dip=10)


def normalize(profile):
    if not isinstance(profile, dict) or set(profile) != set(DEFAULTS) or profile.get('format') != DEFAULTS['format']:
        raise ValueError('网页运动参数记录不完整。')
    result = dict(profile)
    for key, value in profile.items():
        if key in ('joint_acceleration', 'linear_acceleration'):
            # Migrate formerly saved web caps (1..20) to the user's requested
            # inherited mode. Never send AccJ(0), AccL(0), or a reset to 100.
            if type(value) is not int or not 0 <= value <= 20:
                raise ValueError('网页加速度记录无效：' + key)
            result[key] = 0
        elif key != 'format' and (type(value) is not int or not 1 <= value <= DEFAULTS[key]):
            raise ValueError('网页运动参数超出当前试机上限：' + key)
    return result


def apply_to_artwork(artwork, profile):
    """Apply after original artwork/site validation, without editing base strokes."""
    from color_profile import ColorProfile
    ink, plan, strokes, meta = copy.deepcopy(artwork)
    profile = normalize(profile)
    for key, limit in CONTACT_SPEED_LIMITS.items():
        profile[key] = min(profile[key], limit)
    for target in (ink['meta'], meta):
        # Discard the former shared web overlay, including old saved copies.
        # Black/title drawing boost is applied by web.execution at waypoint
        # generation; retain source strokes and other per-artwork settings.
        target.pop('web_motion_profile', None)
        target.pop('runtime_acceleration', None)
        # Used by black/title for cup approach/return and vertical clearance.
        # Per-stroke horizontal travel over paper remains as authored.
        target['air_transfer_velocity_percent'] = min(target.get('air_transfer_velocity_percent', 40),
                                                    profile['horizontal'])
    module = plan['modules']['color']
    candidate = module['motion_candidate']
    travel = candidate['travel_velocity_percent']
    for key, limit, fallback in (
            ('global_speed_factor', 'global_speed_factor', None),
            ('travel_velocity_percent', 'approach', None),
            ('horizontal_velocity_percent', 'horizontal', travel),
            ('approach_velocity_percent', 'approach', travel),
            ('touchdown_velocity_percent', 'touchdown', travel),
            ('lift_velocity_percent', 'lift', travel),
            ('dip_velocity_percent', 'dip', None)):
        source = candidate.get(key)
        candidate[key] = min(fallback if source is None else source, profile[limit])
    candidate['accel_joint_percent'] = 0
    candidate['accel_linear_percent'] = 0
    candidate['acceleration_required'] = False
    candidate['draw_velocity_percent'] = min(candidate['draw_velocity_percent'], profile['draw'])
    candidate['draw_velocity_max'] = min(candidate['draw_velocity_max'], profile['draw'])
    from web.execution import paper_draw_scale
    draw_scale = paper_draw_scale(plan)
    old_draw = candidate['draw_velocity_percent']
    old_cap = candidate['draw_velocity_max']
    for stroke in module.get('strokes', []):
        speeds = stroke.get('speed')
        if speeds is not None:
            # Match the executor's previous 5..cap clamp before increasing.
            stroke['speed'] = [min(100, max(5, min(v, old_cap)) * draw_scale)
                               for v in speeds]
    candidate['draw_velocity_percent'] = min(100, round(old_draw * draw_scale))
    candidate['draw_velocity_max'] = min(100, round(old_cap * draw_scale))
    module['motion_candidate'] = ColorProfile(**candidate).to_dict()
    module['motion_diagnostics'] = True
    return ink, plan, strokes, meta
