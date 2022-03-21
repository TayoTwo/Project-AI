using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PathManager : MonoBehaviour{

    public Transform car;
    public Transform target;
    public List<Cell> openSet = new List<Cell>();
    public HashSet<Cell> closedSet = new HashSet<Cell>();
    CellManager cellManager;
    TargetManager targetManager;

    void Awake(){

        cellManager = GetComponent<CellManager>();
        targetManager = GetComponent<TargetManager>();

        cellManager.Initialize();
        //FindPath(car.position,target.position);

    }

    void Update(){

        FindPath(car.position,target.position);

    }

    void FindPath(Vector3 s, Vector3 e){

        targetManager.ClearTargets();
        openSet.Clear();
        closedSet.Clear();

        Cell start = cellManager.WorldPosToCell(s);
        Cell end = cellManager.WorldPosToCell(e);

        //Debug.Log(end.gridPos);

        openSet.Add(start);

        while(openSet.Count > 0){

            Cell currentCell = openSet[0];

            for(int i = 1; i < openSet.Count;i++){


                //If the open set and closed set have the same F cost then compare their H costs instead (distance to target Cell)
                if(openSet[i].F() < currentCell.F() 
                || (openSet[i].F() == currentCell.F() && openSet[i].h < currentCell.h)){

                                
                    currentCell = openSet[i];

                }

            }

            openSet.Remove(currentCell);
            closedSet.Add(currentCell);

            if(currentCell == end){

                //We've reached our target
                //Debug.Log("PATH FOUND");
                Retrace(start,end);
                targetManager.SetCarTarget();
                return;

            }


            foreach(Cell neigh in cellManager.GetNeighbours(currentCell)){

                if(!neigh.walkable || closedSet.Contains(neigh)){

                    continue;

                }



                int disToNeighbour = currentCell.g + GetDis(currentCell,neigh);

                if(disToNeighbour < neigh.g || !openSet.Contains(neigh)){

                    neigh.g = disToNeighbour;
                    neigh.h = GetDis(neigh,end);
                    neigh.parent = currentCell;

                    if(!openSet.Contains(neigh)){

                        openSet.Add(neigh);

                    }

                }

            }

        }


    }

    void Retrace(Cell start,Cell end){

        List<Cell> path = new List<Cell>();

        Cell current = end;

        //Loop backwards
        while(current != start){
            path.Add(current);
            //Spawn a target at the Cells position
            targetManager.SpawnTarget(current.pos);

            //Set the current Cell to its parent Cell
            current = current.parent;

        }

        path.Reverse();
        targetManager.targets.Reverse();

    }

    int GetDis(Cell a, Cell b){

        int x = Mathf.Abs(a.gridPos.x - b.gridPos.x);
        int y = Mathf.Abs(a.gridPos.y - b.gridPos.y);

        if(x > y){

            return 14 * y + 10 * (x-y);

        } else {

            return 14 * x + 10 * (y-x);

        }
         
    }

}
