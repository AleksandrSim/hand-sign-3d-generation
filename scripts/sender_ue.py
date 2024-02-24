import time
import queue
import multiprocessing

import requests

FPS = 30
FRAME_TIME = 1.0 / FPS
TIMEOUT = 0.1


def generate_linear_interpolation(
        extr: tuple[float, float, float],
        N: int) -> list[list[float]]:
    # Calculate the step size for interpolation
    steps = [x / (N // 2) for x in extr]

    values = [[step * i for step in steps] for i in range(N // 2)]
    values += [[extr[n] - steps[n] * i for n in range(len(extr))]
               for i in range(N // 2 + 1)]

    return values


def send_data(m_queue: multiprocessing.Queue, url: str):
    while True:
        try:
            if not m_queue.empty():
                start_t = time.perf_counter()
                s = m_queue.get(timeout=TIMEOUT)

                requests.put(url, json={
                    "Parameters": {
                        "Thumb 01 R Ctrl": {
                            "Pitch": s[0],
                            "Yaw": s[1],
                            "Roll": s[2]
                        },
                        "Thumb 02 R Ctrl": {
                            "Pitch": 0.0,
                            "Yaw": 0.0,
                            "Roll": 0.0
                        },
                        "Thumb 03 R Ctrl": {
                            "Pitch": s[0],
                            "Yaw": s[1],
                            "Roll": s[2]
                        },
                        "Index Metacarpal R Ctrl": {
                            "Pitch": 0.0,
                            "Yaw": 0.0,
                            "Roll": 0.0
                        },
                        "Index 01 R Ctrl": {
                            "Pitch": s[0],
                            "Yaw": s[1],
                            "Roll": s[2]
                        },
                        "Index 02 R Ctrl": {
                            "Pitch": 0.0,
                            "Yaw": 0.0,
                            "Roll": 0.0
                        },
                        "Index 03 R Ctrl": {
                            "Pitch": 0.0,
                            "Yaw": 0.0,
                            "Roll": 0.0
                        },
                        "Middle Metacarpal R Ctrl": {
                            "Pitch": 0.0,
                            "Yaw": 0.0,
                            "Roll": 0.0
                        },
                        "Middle 01 R Ctrl": {
                            "Pitch": 0.0,
                            "Yaw": 0.0,
                            "Roll": 0.0
                        },
                        "Middle 02 R Ctrl": {
                            "Pitch": 0.0,
                            "Yaw": 0.0,
                            "Roll": 0.0
                        },
                        "Middle 03 R Ctrl": {
                            "Pitch": 0.0,
                            "Yaw": 0.0,
                            "Roll": 0.0
                        },
                        "Ring Metacarpal R Ctrl": {
                            "Pitch": 0.0,
                            "Yaw": 0.0,
                            "Roll": 0.0
                        },
                        "Ring 01 R Ctrl": {
                            "Pitch": 0.0,
                            "Yaw": 0.0,
                            "Roll": 0.0
                        },
                        "Ring 02 R Ctrl": {
                            "Pitch": 0.0,
                            "Yaw": 0.0,
                            "Roll": 0.0
                        },
                        "Ring 03 R Ctrl": {
                            "Pitch": 0.0,
                            "Yaw": 0.0,
                            "Roll": 0.0
                        },
                        "Pinky Metacarpal R Ctrl": {
                            "Pitch": 0.0,
                            "Yaw": 0.0,
                            "Roll": 0.0
                        },
                        "Pinky 01 R Ctrl": {
                            "Pitch": 0.0,
                            "Yaw": 0.0,
                            "Roll": 0.0
                        },
                        "Pinky 02 R Ctrl": {
                            "Pitch": 0.0,
                            "Yaw": 0.0,
                            "Roll": 0.0
                        },
                        "Pinky 03 R Ctrl": {
                            "Pitch": 0.0,
                            "Yaw": 0.0,
                            "Roll": 0.0
                        }
                    },
                    "generateTransaction": True
                },
                    timeout=1.0)
                send_time = time.perf_counter() - start_t
                time.sleep(max(FRAME_TIME-send_time, 0.0))
            else:
                values = generate_linear_interpolation((10., 50., 120.), 100)
                for val in values:
                    m_queue.put(val)
        except queue.Empty:  # Here queue is a mudule, not a class entity
            pass
        except Exception as e:
            print(f"Error occurred while sending data: {e}")


def listen_input(m_queue: multiprocessing.Queue):
    while True:
        try:
            user_input = input("Enter a string: ")
            for char in user_input:
                m_queue.put(char)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    # Define the URL where characters will be sent
    # endpoint_url = "http://127.0.0.1:8000"
    endpoint_url = "http://localhost:30010/remote/preset/RCP_Hands/function/Set%20Hand%20Pose"

    # Create a multiprocessing queue for communication between processes
    data_m_queue = multiprocessing.Queue()

    # TODO Populate the queue with actual values

    # Create and start the processes
    sender_process = multiprocessing.Process(
        target=send_data, args=(data_m_queue, endpoint_url))
    sender_process.start()

    try:
        # Join the processes to wait for their completion
        # input_process.join()
        sender_process.join()
    except KeyboardInterrupt:
        # If the user interrupts the program, terminate the processes
        # input_process.terminate()
        sender_process.terminate()
