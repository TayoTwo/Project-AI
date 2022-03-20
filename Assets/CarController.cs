using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CarController : MonoBehaviour
{
    [Header("Targeting")]
    public Transform target;
    public float maxAngle;
    public float steerSpeed;
    public float torque;

    [Header("Wheel Info")]
    public WheelCollider[] wheels;
    public Transform[] wheelModels;

    // Start is called before the first frame update
    void Start()
    {
        
    }

    void FixedUpdate(){

        AddTorque(torque);
        WheelTargeting();

    }

    // Update is called once per frame
    void Update()
    {
        UpdateWheelsPos();
    }

    void AddTorque(float t){

        for(int i = 0; i < wheels.Length;i++){

            wheels[i].motorTorque = torque;

        }

    }

    void WheelTargeting(){

        float steerAngle = Vector3.SignedAngle(transform.forward,target.position,Vector3.up);

        steerAngle = Mathf.Clamp(steerAngle,-maxAngle,maxAngle);

        wheels[2].steerAngle = steerAngle;
        wheels[3].steerAngle = steerAngle;

    }

    void UpdateWheelsPos(){

        Vector3 pos;
        Quaternion rot;

        for(int i = 0; i < wheels.Length;i++){

            wheels[i].GetWorldPose(out pos,out rot);
            wheelModels[i].position = pos;
            wheelModels[i].rotation = rot;

        }

    }
}
