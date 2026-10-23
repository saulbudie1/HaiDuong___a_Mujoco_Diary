# HaiDuong

<div align="center">

### Latest test: [Test 8](Tests/Test8/Test8.md)
**Rough terrain, about 91.75M steps (the run stopped early): the Go1 follows straight, diagonal, circling and fast-spin commands on bumpy ground.**

<a href="Tests/Test8/Test8.md"><img src="Tests/Test8/reward_plot.png" width="560" alt="Test 8, eval reward vs steps"></a>
<br><sub>eval reward vs steps &nbsp;·&nbsp; 24.1 at 91.75M steps (stopped early, GPU out of memory)</sub>

<a href="Tests/Test8/Test8.md"><img src="Tests/Test8/preview/top.gif" width="320" alt="Test 8, top view"></a>
<br><sub>top view &nbsp;·&nbsp; command x = 0.5, y = 0.5, yaw = 3 &nbsp;·&nbsp; mean speed tracking 0.848 (walking in a tight circle)</sub>
<br><sub>full videos: <a href="Tests/Test8/x0.5_y0.5_yaw3.0/side_x0.5_y0.5_yaw3.0_ckpt_91750400.mp4">side</a> &nbsp;·&nbsp; <a href="Tests/Test8/x0.5_y0.5_yaw3.0/top_x0.5_y0.5_yaw3.0_ckpt_91750400.mp4">top</a> &nbsp;·&nbsp; <a href="Tests/Test8/x0.5_y0.5_yaw3.0/track_x0.5_y0.5_yaw3.0_ckpt_91750400.mp4">track</a></sub>

<a href="Tests/Test8/Test8.md"><img src="Tests/Test8/x0.5_y0.5_yaw3.0/motion_x0.5_y0.5_yaw3.0_ckpt_91750400.png" width="560" alt="Test 8, distance, velocity and rotation for command 0.5, 0.5, 3"></a>
<br><sub>distance, velocity vs command and rotation &nbsp;·&nbsp; command x = 0.5, y = 0.5, yaw = 3</sub>

<a href="Tests/Test8/Test8.md"><img src="Tests/Test8/x0.5_y0.5_yaw3.0/trail_x0.5_y0.5_yaw3.0_ckpt_91750400.png" width="480" alt="Test 8, bird's-eye trail for command 0.5, 0.5, 3"></a>
<br><sub>bird's-eye trail: body centre and head tip, start to end &nbsp;·&nbsp; command x = 0.5, y = 0.5, yaw = 3 &nbsp;·&nbsp; about 9 laps in 20 s</sub>

<br>

<table align="center">
<tr>
<td align="center" width="560">
<a href="Declaration.md"><b>What I use and where it comes from</b></a>
<br><sub>libraries and versions &nbsp;·&nbsp; Matplotlib, pandas &nbsp;·&nbsp; MuJoCo Playground, Brax, JAX &nbsp;·&nbsp; Go1 model &nbsp;·&nbsp; WSL2 Ubuntu &nbsp;·&nbsp; use of AI &nbsp;·&nbsp; references</sub>
</td>
</tr>
</table>

</div>

<!-- Keep the fun fact as the last section of this README. New sections go above it. -->

> **Fun fact:** HaiDuong (Hải Dương) is my hometown in Vietnam. I have named the repo after it
> because it was windy this day and it reminds me of the autumn there.
