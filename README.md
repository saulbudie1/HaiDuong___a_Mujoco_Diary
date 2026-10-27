# HaiDuong

<div align="center">

### Latest test: [Test 9.3](Tests/Test9.3/Test9.3.md)
**The Test 7 policy, trained on flat ground only, walks across a new 100 m sine-wave ground with speed tracking 0.92 (0.94 on flat ground), here wearing a visual-only cone hat. No training on this ground yet.**

<a href="Tests/Test9.3/Test9.3.md"><img src="Tests/Test9.3/preview/track_cone.gif" width="288" alt="Test 9.3 with the cone hat, tracking-camera view"></a>
<a href="Tests/Test9.3/Test9.3.md"><img src="Tests/Test9.3/preview/top_cone.gif" width="288" alt="Test 9.3 with the cone hat, top view"></a>
<br><sub>tracking view and top view &nbsp;·&nbsp; command x = 1 (forward 1 m/s) &nbsp;·&nbsp; mean speed tracking 0.920 &nbsp;·&nbsp; with the cone hat (no effect on the physics) &nbsp;·&nbsp; first 6 s of a 20 s run</sub>
<br><sub>full videos: <a href="Tests/Test9.3/x1.0_y0.0_yaw0.0/side_x1.0_y0.0_yaw0.0_SimpleUniFormRoughGround_1_go1_final_cone.mp4">side</a> &nbsp;·&nbsp; <a href="Tests/Test9.3/x1.0_y0.0_yaw0.0/top_x1.0_y0.0_yaw0.0_SimpleUniFormRoughGround_1_go1_final_cone.mp4">top</a> &nbsp;·&nbsp; <a href="Tests/Test9.3/x1.0_y0.0_yaw0.0/track_x1.0_y0.0_yaw0.0_SimpleUniFormRoughGround_1_go1_final_cone.mp4">track</a></sub>

<a href="Tests/Test9.3/Test9.3.md"><img src="Tests/Test9.3/x1.0_y0.0_yaw0.0/motion_x1.0_y0.0_yaw0.0_SimpleUniFormRoughGround_1_go1_final_cone.png" width="560" alt="Test 9.3, distance, velocity and rotation for command 1, 0, 0"></a>
<br><sub>distance, velocity vs command and rotation &nbsp;·&nbsp; command x = 1, y = 0, yaw = 0 &nbsp;·&nbsp; 18.6 m in 20 s</sub>

<a href="Tests/Test9.3/Test9.3.md"><img src="Tests/Test9.3/x1.0_y0.0_yaw0.0/trail_x1.0_y0.0_yaw0.0_SimpleUniFormRoughGround_1_go1_final_cone.png" width="480" alt="Test 9.3, bird's-eye trail for command 1, 0, 0"></a>
<br><sub>bird's-eye trail: body centre and head tip (drawn 2 m ahead), start to end &nbsp;·&nbsp; a nearly straight 18.6 m line</sub>

<br>

<table align="center">
<tr>
<td align="center" width="560">
<a href="Declaration.md"><b>What I use and where it comes from</b></a>
<br><sub>libraries and versions &nbsp;·&nbsp; Matplotlib, pandas &nbsp;·&nbsp; MuJoCo Playground, Brax, JAX &nbsp;·&nbsp; Go1 model &nbsp;·&nbsp; WSL2 Ubuntu &nbsp;·&nbsp; use of AI &nbsp;·&nbsp; references</sub>
</td>
</tr>
</table>

<br>

<a href="Tests/Test9.3/Test9.3.md"><img src="Assets/Grounds/SimpleUniFormRoughGround_1_preview_closeup.png" width="560" alt="The new sine-wave ground with a Go1-sized stand-in"></a>
<br><sub>the new ground: two crossing sine waves, 0.4 m wavelength, crest heights from 1x to 1.5x, 100 x 100 m &nbsp;·&nbsp; the red box is a Go1-sized stand-in for scale &nbsp;·&nbsp; written by Claude, see <a href="Declaration.md">Declaration</a></sub>

</div>

<!-- Keep the fun fact as the last section of this README. New sections go above it. -->

> **Fun fact:** HaiDuong (Hải Dương) is my hometown in Vietnam. I have named the repo after it
> because it was windy this day and it reminds me of the autumn there.
