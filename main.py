import argparse
import pickle
from datetime import datetime
from pathlib import Path
from time import perf_counter

import coal
import numpy as np
import pinocchio as pin
import yaml
from loguru import logger
from scipy import interpolate

import opt_prob
import opt_term
from utils.logger import setup_logger
from utils.robot import compute_traj_ik, load_robot


def set_seed(seed: int):
    pin.seed(seed)
    coal.seed(seed)
    np.random.seed(seed)


def load_all_traj_data(data_path: str):
    data_dict = pickle.load(open(data_path, "rb"))
    data_dict = dict(filter(lambda x: "traj" in x[0], data_dict.items()))
    data_dict = dict(
        sorted(
            data_dict.items(),
            key=lambda x: int(x[0].split("_")[-1]) if x[0] != "traj_test" else -1,
        )
    )
    return data_dict


def parse_traj_data(
    traj_data: dict,
    model: pin.Model,
    ee_frame_id: int,
    num_timesteps: int,
    rng_gen: np.random.Generator,
):
    T_ee2world_list = []
    for quat_wxyz, translation in zip(
        traj_data["traj_quaternions"], traj_data["traj_positions"]
    ):
        T_ee2world = pin.SE3(
            pin.Quaternion(quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]),
            np.array(translation),
        )
        T_ee2world_list.append(T_ee2world)

    traj_q_init = compute_traj_ik(model, ee_frame_id, T_ee2world_list, rng_gen=rng_gen)

    if num_timesteps is None:
        traj_q_init = np.array(traj_q_init)
        timestep_T_ee2world_list = np.arange(len(T_ee2world_list))
    else:
        assert num_timesteps >= len(T_ee2world_list)
        if num_timesteps == len(T_ee2world_list):
            traj_q_init = np.array(traj_q_init)
            timestep_T_ee2world_list = np.arange(len(T_ee2world_list))
        else:
            # Assume the timestamps are uniformly distributed
            traj_q_init = interpolate.interp1d(
                np.arange(0, len(T_ee2world_list)) / (len(T_ee2world_list) - 1),
                np.array(traj_q_init),
                axis=0,
                kind="linear",
            )(np.linspace(0, 1, num_timesteps, endpoint=True))
            timestep_T_ee2world_list = (
                np.round(
                    (
                        np.linspace(0, 1, len(T_ee2world_list), endpoint=True)
                        * (num_timesteps - 1)
                    )
                )
                .astype(np.int64)
                .clip(0, num_timesteps - 1)
            )
    T_ee2world_list = np.array(T_ee2world_list)
    timestep_T_ee2world_list = [
        timestep_T_ee2world for timestep_T_ee2world in timestep_T_ee2world_list
    ]
    T_ee2world_list = [T_ee2world for T_ee2world in T_ee2world_list]
    return traj_q_init, timestep_T_ee2world_list, T_ee2world_list


def calc_statistics(results: dict):
    num_success = 0
    cost_time = []
    num_tries = []
    for result in results.values():
        if result["status"] == opt_prob.ProbStatus.Converged.value:
            num_success += 1
            cost_time.append(result["cost_time"])
            num_tries.append(result["num_tries"])

    return {
        "success_rate": num_success / len(results) * 100.0,
        "failure_rate": (1 - num_success / len(results)) * 100.0,
        "cost_time_avg": np.mean(cost_time) if len(cost_time) > 0 else 0.0,
        "cost_time_std": np.std(cost_time) if len(cost_time) > 0 else 0.0,
        "num_tries_avg": np.mean(num_tries) if len(num_tries) > 0 else 0.0,
        "num_tries_std": np.std(num_tries) if len(num_tries) > 0 else 0.0,
    }


def construct_problem(
    model: pin.Model,
    collision_model: pin.GeometryModel,
    ee_frame_id: int,
    traj_q_init: np.ndarray,
    timestep_T_ee2world_list: list[int],
    T_ee2world_list: list[np.ndarray],
    opt_cfg: dict,
    with_self_collision: bool = True,
):
    traj_q_lower_bound = model.lowerPositionLimit.reshape(1, -1).repeat(
        traj_q_init.shape[0], axis=0
    )
    traj_q_upper_bound = model.upperPositionLimit.reshape(1, -1).repeat(
        traj_q_init.shape[0], axis=0
    )

    trust_region_size = np.abs(traj_q_upper_bound - traj_q_lower_bound) * 0.01
    min_trust_region_size = np.abs(traj_q_upper_bound - traj_q_lower_bound) * 0.0001

    prob = opt_prob.TR_SP(**opt_cfg["tr_sp"])

    prob.set_variable(traj_q_init, traj_q_lower_bound, traj_q_upper_bound)
    prob.set_trust_region_size(trust_region_size, min_trust_region_size)

    prob.add_term(
        opt_term.TrajSmoothTerm(
            k=opt_cfg["traj_smooth_term"]["k"],
            offset=opt_cfg["traj_smooth_term"]["offset"],
            threshold=opt_cfg["traj_smooth_term"]["threshold"],
        ),
        merit_coeff=opt_cfg["traj_smooth_term"]["merit_coeff"],
    )
    prob.add_term(
        opt_term.BaseFixationTerm(
            threshold=opt_cfg["base_fixation_term"]["threshold"],
        ),
        merit_coeff=opt_cfg["base_fixation_term"]["merit_coeff"],
    )
    prob.add_term(
        opt_term.EEPoseTerm(
            model,
            ee_frame_id,
            timestep_T_ee2world_list,
            T_ee2world_list,
            threshold=opt_cfg["ee_pose_term"]["threshold"],
        ),
        merit_coeff=opt_cfg["ee_pose_term"]["merit_coeff"],
    )
    if with_self_collision:
        prob.add_term(
            opt_term.DiscreteCollisionTerm(
                model,
                collision_model,
                sd_check=opt_cfg["self_collision_term"]["sd_check"],
                sd_safe=opt_cfg["self_collision_term"]["sd_safe"],
                threshold=opt_cfg["self_collision_term"]["threshold"],
            ),
            merit_coeff=opt_cfg["self_collision_term"]["merit_coeff"],
        )
    return prob


def main(args):
    args.log_dir = (
        Path(args.log_dir)
        / Path(args.traj_data_path).stem
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    args.log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(args.log_level, log_dir=args.log_dir)
    logger.log(args.log_level, vars(args))

    opt_cfg = yaml.safe_load(open(args.opt_cfg_path, "r"))
    logger.log(args.log_level, opt_cfg)

    data_dict = load_all_traj_data(args.traj_data_path)
    logger.log(args.log_level, f"File path: {args.traj_data_path}")

    model, collision_model, ee_frame_id = load_robot(
        args.robot_cfg_path,
        args.asset_path,
        with_ground_plane=not args.disable_ground_plane,
    )
    assert np.all(model.hasConfigurationLimit())

    results = {}
    for idx, (traj_name, traj_data) in enumerate(data_dict.items()):
        logger.log(
            args.log_level, f"[{idx + 1} / {len(data_dict)}] Processing {traj_name}..."
        )

        seed = opt_cfg["seed"] + idx
        set_seed(seed)
        rng_gen = np.random.default_rng(seed=seed)

        start_time = perf_counter()
        num_tries = 0
        while num_tries < opt_cfg["max_tries"]:
            num_tries += 1

            traj_q_init, timestep_T_ee2world_list, T_ee2world_list = parse_traj_data(
                traj_data, model, ee_frame_id, args.num_timesteps, rng_gen=rng_gen
            )
            logger.success(
                f"Succeeded to initialize variable for {traj_name} after {num_tries} tries."
            )

            if not args.disable_initial_guess:
                prob_initial_guess = construct_problem(
                    model,
                    collision_model,
                    ee_frame_id,
                    traj_q_init,
                    timestep_T_ee2world_list,
                    T_ee2world_list,
                    opt_cfg,
                    with_self_collision=False,
                )
                result = prob_initial_guess.solve(
                    solver="COPT",
                    solver_verbose=args.solver_verbose,
                    ret_all_steps=args.ret_all_steps,
                )

                if result.status == opt_prob.ProbStatus.Converged:
                    logger.success(
                        f"Succeeded to solve the initial guess problem for {traj_name} after {num_tries} tries."
                    )
                else:
                    logger.error(
                        f"Failed to solve the initial guess problem for {traj_name} after {num_tries} tries."
                    )
                    logger.warning("Retrying...")
                    continue

                traj_q_init = result.variable

            prob = construct_problem(
                model,
                collision_model,
                ee_frame_id,
                traj_q_init,
                timestep_T_ee2world_list,
                T_ee2world_list,
                opt_cfg,
                with_self_collision=True,
            )
            result = prob.solve(
                solver="COPT",
                solver_verbose=args.solver_verbose,
                ret_all_steps=args.ret_all_steps,
            )

            if result.status == opt_prob.ProbStatus.Converged:
                logger.success(
                    f"Succeeded to solve the problem for {traj_name} after {num_tries} tries."
                )
                break
            else:
                logger.error(
                    f"Failed to solve the problem for {traj_name} after {num_tries} tries."
                )
                logger.warning("Retrying...")
                continue

        end_time = perf_counter()
        cost_time = end_time - start_time

        results[traj_name] = {
            "status": result.status.value,
            "cost_time": cost_time,
            "num_tries": num_tries,
            "variable": result.variable,
            "cost_values": result.cost_values,
            "constraint_violations": result.constraint_violations,
        }
        if args.ret_all_steps:
            results[traj_name]["variable_all_steps"] = result.variable_all_steps

        statistics = calc_statistics(results)
        logger.log(
            args.log_level,
            "\n========== Statistics ==========\n"
            f"Progress: {idx + 1} / {len(data_dict)}\n"
            f"Success rate: {statistics['success_rate']:.2f}%\n"
            f"Failure rate: {statistics['failure_rate']:.2f}%\n"
            f"Cost time: {statistics['cost_time_avg']:.2f}s "
            f"± {statistics['cost_time_std']:.2f}s\n"
            f"Num tries: {statistics['num_tries_avg']:.2f} "
            f"± {statistics['num_tries_std']:.2f}\n",
        )

    with open(Path(args.log_dir) / "results.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opt_cfg_path",
        type=str,
        default=(Path(__file__).parent / "opt_cfg" / "default.yml").as_posix(),
    )
    parser.add_argument("--robot_cfg_path", type=str, required=True)
    parser.add_argument(
        "--asset_path",
        type=str,
        default=(Path(__file__).parent / "content" / "assets").as_posix(),
    )
    parser.add_argument("--traj_data_path", type=str, required=True)
    parser.add_argument("--num_timesteps", type=int, default=None)
    parser.add_argument("--disable_ground_plane", action="store_true", default=False)
    parser.add_argument("--disable_initial_guess", action="store_true", default=False)
    parser.add_argument("--ret_all_steps", action="store_true", default=False)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument(
        "--log_dir", type=str, default=(Path(__file__).parent / "logs").as_posix()
    )
    parser.add_argument("--solver_verbose", action="store_true")

    args = parser.parse_args()
    main(args)
