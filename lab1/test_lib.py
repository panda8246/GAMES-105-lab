from scipy.spatial.transform import Rotation as R

a = R.from_rotvec([0, 3, 2])
b = R.from_rotvec([2, 1, 5])
c = R.from_rotvec([4, 7, 3])
ans1 = (a * b * c).as_matrix()
ans2 = a.as_matrix() @ b.as_matrix() @ c.as_matrix()
# print(ans1, ans2)
ee = [None] * 5
ee[2:6] = 1, 2, 3
print(ee)

