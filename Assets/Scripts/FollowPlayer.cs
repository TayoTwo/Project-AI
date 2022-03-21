using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class FollowPlayer : MonoBehaviour
{

    public Transform target;
    public Vector3 offset;
    public float smoothTime;
    Vector3 vel;

    void LateUpdate(){

        Vector3 posToMove = target.position + offset;
        transform.position = Vector3.SmoothDamp(transform.position,posToMove,ref vel,smoothTime);
        transform.LookAt(target.position);
        
    }

}
